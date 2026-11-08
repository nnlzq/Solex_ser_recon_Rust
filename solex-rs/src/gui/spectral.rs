//! Spectral Analyser window (port of spectralAnalyserUI.py).

use super::resources;
use super::view;
use crate::edge;
use crate::fitline;
use crate::geom;
use crate::num::{Img, ImgF};
use crate::pipeline;
use crate::pngio;
use crate::procss::{self, Options};
use crate::recon;
use crate::ser::SerInfo;
use crate::vig;
use egui::Color32;
use serde_json::{Map, Value};

pub struct Env {
    pub output_dir: String,
    pub ratio_fixe: Option<f64>,
    pub slant_fix: Option<f64>,
    pub ellipse_fit_shift: i64,
    pub flip_x: bool,
    pub de_vignette: bool,
    pub dispersion: f64,
    pub spec_dir: String,
    pub transversalium: bool,
    pub trans_strength: i64,
    pub stubborn: bool,
    pub img_rotate: i64,
    pub delta_radius: i64,
    pub fixed_width: Option<i64>,
    pub crop_square: bool,
}

struct Loaded {
    info: SerInfo,
    file: String,
    mean: Img,
    fit: Vec<(i64, f64, f64)>,
    curve: Vec<f64>,
    ratio: Option<f64>,
    slant: Option<f64>,
    borders: [f64; 4],
    bounds: (usize, usize),
    cercle: (f64, f64, f64),
    f: f64,
}

pub enum Action {
    None,
    Close(Option<i64>),
}

/// Status line content; kept as a key so it re-translates on language switch.
enum Status {
    Key(&'static str),
    Dispersion(f64),
}

pub struct Spectral {
    pub open: bool,
    lang_idx: usize,
    dict: Map<String, Value>,
    env: Env,
    file_text: String,
    anchors: (Vec<f64>, Vec<String>, Vec<String>),
    targets: (Vec<f64>, Vec<String>, Vec<String>),
    alps: (f64, f64, Vec<u8>),
    anchor_idx: Option<usize>,
    target_idx: Option<usize>,
    ashift_text: String,
    disp_text: String,
    shift_text: String,
    shift: i64,
    dispersion: Option<f64>,
    wavelength_text: String,
    status: Status,
    popup: Option<String>,
    loaded: Option<Loaded>,
    cc_tex: Option<egui::TextureHandle>,
    protus_tex: Option<egui::TextureHandle>,
    mean_tex: Option<egui::TextureHandle>,
}

impl Spectral {
    pub fn new(env: Env, lang_idx: usize, dict: Map<String, Value>) -> Spectral {
        let mut sp = Spectral {
            open: true,
            lang_idx,
            dict,
            file_text: env.spec_dir.clone(),
            anchors: resources::anchor_lines(),
            targets: resources::target_lines(),
            alps: resources::alps(),
            anchor_idx: None,
            target_idx: None,
            ashift_text: "0".into(),
            disp_text: format!("{}", env.dispersion),
            shift_text: "0".into(),
            shift: 0,
            dispersion: None,
            wavelength_text: String::new(),
            status: Status::Key("Choose a file and press Start analysis."),
            popup: None,
            env,
            loaded: None,
            cc_tex: None,
            protus_tex: None,
            mean_tex: None,
        };
        sp.wavelength_text = sp.wavelength_none();
        sp
    }

    /// Keep the window labels in sync when the main window changes language.
    pub fn sync_lang(&mut self, lang_idx: usize, dict: &Map<String, Value>) {
        if self.lang_idx != lang_idx {
            self.lang_idx = lang_idx;
            self.dict = dict.clone();
            self.update_wavelength();
        }
    }

    fn t(&self, key: &str) -> String {
        match self.dict.get(key).and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => key.to_string(),
        }
    }

    fn wavelength_none(&self) -> String {
        format!("{}: {}", self.t("Wavelength shift"), self.t("None"))
    }

    fn status_text(&self) -> String {
        match self.status {
            Status::Key(k) => self.t(k),
            Status::Dispersion(d) => format!("{} {:.6}", self.t("the dispersion is:"), d),
        }
    }

    fn analyze(&mut self, ctx: &egui::Context) {
        self.loaded = None;
        self.cc_tex = None;
        self.protus_tex = None;
        self.mean_tex = None;
        self.shift = 0;
        self.shift_text = "0".into();
        self.wavelength_text = self.wavelength_none();
        let file = self.file_text.split(';').next().unwrap_or("").trim().to_string();
        if file.is_empty() {
            self.popup = Some(self.t("Error: no file selected"));
            return;
        }
        let info = match SerInfo::probe(&file) {
            Ok(i) => i,
            Err(e) => {
                self.popup = Some(format!("{}: {}", self.t("Error"), e));
                return;
            }
        };
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let (mean, maxi) = match fitline::compute_mean_max(&file, &info, threads) {
            Ok(v) => v,
            Err(e) => {
                self.popup = Some(format!("{}: {}", self.t("Error"), e));
                return;
            }
        };
        let lf = fitline::compute_mean_return_fit(&file, &info, &mean, &maxi, threads);
        let bounds = (lf.y1, lf.y2);
        let mut ratio = self.env.ratio_fixe;
        let mut slant = self.env.slant_fix;
        let mut borders = [0.0f64; 4];
        let mut cercle = (-1.0f64, -1.0f64, -1.0f64);
        if ratio.is_none() && slant.is_none() {
            match recon::reconstruct(&file, &info, &lf.fit, self.env.ellipse_fit_shift, threads) {
                Ok(disk) => {
                    let disk = if self.env.flip_x {
                        disk_img(&info, disk).flip_x()
                    } else {
                        disk_img(&info, disk)
                    };
                    let res = edge::ellipse_to_circle(&disk);
                    ratio = Some(res.ratio);
                    slant = Some(res.phi.to_degrees());
                    borders = res.borders;
                    cercle = res.circle;
                }
                Err(e) => {
                    self.popup = Some(format!("{}: {}", self.t("Error"), e));
                    return;
                }
            }
        }
        let factor = (1000.0f64.max(info.ih as f64 / 3.0)) / info.ih as f64;
        let loaded = Loaded {
            info,
            file: file.clone(),
            mean,
            fit: lf.fit,
            curve: lf.curve,
            ratio,
            slant,
            borders,
            bounds,
            cercle,
            f: factor,
        };
        self.loaded = Some(loaded);
        self.status = Status::Key("Analysis complete.");
        self.refresh(ctx);
    }

    fn refresh(&mut self, ctx: &egui::Context) {
        let Some(l) = &self.loaded else { return };
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let disk = match recon::reconstruct(&l.file, &l.info, &l.fit, self.shift, threads) {
            Ok(d) => d,
            Err(e) => {
                self.popup = Some(format!("{}: {}", self.t("Error"), e));
                return;
            }
        };
        let disk = if self.env.flip_x { disk_img(&l.info, disk).flip_x() } else { disk_img(&l.info, disk) };
        let small = bilinear_u16(&disk, l.f);
        let ind = ImgF { h: small.h, w: small.w, d: small.d.iter().map(|&v| v as f64 / 65536.0).collect() };
        let phi = l.slant.unwrap_or(0.0).to_radians();
        let ratio = l.ratio.unwrap_or(1.0);
        let (fix_u16, _nc, _m3) = geom::correct_image(&ind, phi, ratio, (-1.0, -1.0), -1.0);
        let cercle = scaled_circle(l.cercle, l.f);
        let borders = scaled_borders(l.borders, l.f);
        let bounds = scaled_bounds(l.bounds, l.f);
        let fix: ImgF = if self.env.de_vignette {
            if cercle.0 == -1.0 {
                self.status = Status::Key("WARNING: cannot de-vignette without ellipse fit");
                fix_u16.to_f()
            } else {
                vig::remove_vignette(&fix_u16, cercle)
            }
        } else {
            fix_u16.to_f()
        };
        let mut o = Options::default();
        o.shift = vec![self.shift];
        o.save_fit = false;
        o.nolog = true;
        o.nthreads = threads;
        o.img_rotate = self.env.img_rotate;
        o.delta_radius = self.env.delta_radius;
        o.transversalium = self.env.transversalium;
        o.trans_strength = self.env.trans_strength;
        o.stubborn = self.env.stubborn;
        o.fixed_width = self.env.fixed_width;
        o.crop_width_square = self.env.crop_square;
        o.disk_display = true;
        o.output_dir = self.env.output_dir.clone();
        if let Some(p) = pipeline::single_image_process(fix, &o, cercle, &borders, "", bounds) {
            self.cc_tex = Some(view::make_texture(ctx, &p.clahe, 900));
            self.protus_tex = Some(view::make_texture(ctx, &p.protus, 900));
        }
        self.mean_tex = Some(view::make_texture(ctx, &l.mean, 1600));
    }

    fn save_image(&mut self) {
        let Some(l) = &self.loaded else { return };
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let disk = match recon::reconstruct(&l.file, &l.info, &l.fit, self.shift, threads) {
            Ok(d) => d,
            Err(e) => {
                self.popup = Some(format!("{}: {}", self.t("Error"), e));
                return;
            }
        };
        let disk = if self.env.flip_x { disk_img(&l.info, disk).flip_x() } else { disk_img(&l.info, disk) };
        let ind = ImgF { h: disk.h, w: disk.w, d: disk.d.iter().map(|&v| v as f64 / 65536.0).collect() };
        let phi = l.slant.unwrap_or(0.0).to_radians();
        let ratio = l.ratio.unwrap_or(1.0);
        let (fix_u16, _nc, _m3) = geom::correct_image(&ind, phi, ratio, (-1.0, -1.0), -1.0);
        let fix: ImgF = if self.env.de_vignette {
            if l.cercle.0 != -1.0 {
                vig::remove_vignette(&fix_u16, l.cercle)
            } else {
                self.status = Status::Key("WARNING: cannot de-vignette without ellipse fit");
                fix_u16.to_f()
            }
        } else {
            fix_u16.to_f()
        };
        let mut o = Options::default();
        o.shift = vec![self.shift];
        o.save_fit = false;
        o.nolog = true;
        o.img_rotate = self.env.img_rotate;
        o.delta_radius = self.env.delta_radius;
        o.transversalium = self.env.transversalium;
        o.trans_strength = self.env.trans_strength;
        o.stubborn = self.env.stubborn;
        o.fixed_width = self.env.fixed_width;
        o.crop_width_square = self.env.crop_square;
        o.disk_display = true;
        o.output_dir = self.env.output_dir.clone();
        let Some(p) = pipeline::single_image_process(fix, &o, l.cercle, &l.borders, "", l.bounds) else {
            return;
        };
        let base = std::path::Path::new(&l.file).with_extension("");
        let basename = format!("{}_shift={}", base.to_string_lossy(), self.shift);
        let _ = pngio::write_png16(&procss::output_path(&(basename.clone() + "_clahe.png"), &o), &p.clahe);
        let _ = pngio::write_png16(&procss::output_path(&(basename + "_protus.png"), &o), &p.protus);
        self.status = Status::Key("Image saved.");
    }

    fn auto_dispersion(&mut self) {
        let Some(l) = &self.loaded else { return };
        let Some(ai) = self.anchor_idx else { return };
        let anchor_guess = self.anchors.0[ai];
        let len = l.mean.w;
        let mid = l.mean.h / 2;
        let spectrum2: Vec<f64> = (0..l.mean.w).map(|x| l.mean.g(mid, x) as f64).collect();
        let anchor_x = l.curve[mid] as i64;
        let (first, step, refy) = &self.alps;
        let n_scales = len * 3;
        let mut best = (f64::NEG_INFINITY, 0.02f64);
        let lspec_full = log_vec(&spectrum2);
        let mean_lspec = mean(&lspec_full);
        for k in 0..n_scales {
            let scale = 0.02 + (0.12 - 0.02) * k as f64 / (n_scales.max(2) - 1) as f64;
            let lx_lo = anchor_guess + (0.0 - anchor_x as f64) * scale;
            let lx_hi = anchor_guess + (len as f64 - anchor_x as f64) * scale;
            let i0 = (((lx_lo - *first) / *step).ceil() as i64).max(0) as usize;
            let i1 = (((lx_hi - *first) / *step).floor() as i64).min(refy.len() as i64 - 1) as usize;
            if i1 <= i0 {
                continue;
            }
            let mut xs: Vec<f64> = Vec::with_capacity(i1 - i0 + 1);
            let mut ys: Vec<f64> = Vec::with_capacity(i1 - i0 + 1);
            for i in i0..=i1 {
                let lx = *first + *step * i as f64;
                let x = (lx - anchor_guess) / scale + anchor_x as f64;
                if x >= 0.0 && x < len as f64 {
                    xs.push(x);
                    ys.push(refy[i] as f64 / 255.0);
                }
            }
            if xs.len() < 2 {
                continue;
            }
            let mut interp = vec![0.0f64; len];
            let mean_ref = mean(&ys);
            for (j, v) in interp.iter_mut().enumerate() {
                *v = interp_linear(j as f64, &xs, &ys);
            }
            let lo = anchor_x.max(0) as usize;
            let hi = (anchor_x + 5).min(len as i64 - 1) as usize;
            for v in interp.iter_mut().take(hi).skip(lo) {
                *v = mean_ref;
            }
            let mut lspec = lspec_full.clone();
            for v in lspec.iter_mut().take(hi).skip(lo) {
                *v = mean_lspec;
            }
            let c = corr(&interp, &lspec);
            if c > best.0 {
                best = (c, scale);
            }
        }
        self.dispersion = Some(best.1);
        self.disp_text = format!("{:.6}", best.1);
        self.status = Status::Dispersion(best.1);
    }

    pub fn ui(&mut self, ctx: &egui::Context) -> Action {
        let mut action = Action::None;
        let mut close_after = false;
        // Native, independent top-level window: its size is not bound by the main window.
        let viewport_id = egui::ViewportId::from_hash_of("spectral_analyser");
        let mut size = egui::vec2(1080.0, 860.0);
        if let Some(mon) = ctx.input(|i| i.viewport().monitor_size) {
            if mon.x > 0.0 {
                size.x = size.x.min(mon.x * 0.94);
            }
            if mon.y > 0.0 {
                size.y = size.y.min(mon.y * 0.92);
            }
        }
        let builder = egui::ViewportBuilder::default()
            .with_title(self.t("Spectral Analyser"))
            .with_inner_size(size)
            .with_min_inner_size([560.0, 400.0])
            .with_resizable(true);
        ctx.show_viewport_immediate(viewport_id, builder, |ctx, _class| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.add_sized([40.0, 20.0], egui::Label::new(self.t("File")));
                    ui.add_sized([420.0, 22.0], egui::TextEdit::singleline(&mut self.file_text));
                    if ui.button(self.t("Choose file")).clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("Video Files (AVI or SER)", &["ser", "avi"])
                            .pick_file()
                        {
                            self.file_text = p.to_string_lossy().into_owned();
                            self.env.spec_dir = p.parent().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
                            self.analyze(ctx);
                        }
                    }
                    if ui.button(self.t("Start analysis")).clicked() {
                        self.analyze(ctx);
                    }
                    if ui.button(self.t("Save image")).clicked() {
                        self.save_image();
                    }
                    if ui.button(self.t("Exit")).clicked() {
                        close_after = true;
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label(self.t("Anchor line"));
                    let prev_anchor = self.anchor_idx;
                    let cur = self.anchor_idx.map(|i| self.anchors.2[i].clone()).unwrap_or_default();
                    let anchor_items = self.anchors.2.clone();
                    egui::ComboBox::from_id_source("anchor")
                        .selected_text(cur)
                        .width(150.0)
                        .show_ui(ui, |ui| {
                            for (i, n) in anchor_items.iter().enumerate() {
                                if ui.selectable_label(self.anchor_idx == Some(i), n).clicked() {
                                    self.anchor_idx = Some(i);
                                }
                            }
                        });
                    if prev_anchor != self.anchor_idx {
                        self.apply_dispersion();
                        self.update_wavelength();
                    }
                    ui.add_space(10.0);
                    ui.label(self.t("GOTO line"));
                    let cur = self.target_idx.map(|i| self.targets.2[i].clone()).unwrap_or_default();
                    let target_items = self.targets.2.clone();
                    egui::ComboBox::from_id_source("target")
                        .selected_text(cur)
                        .width(150.0)
                        .show_ui(ui, |ui| {
                            for (i, n) in target_items.iter().enumerate() {
                                if ui.selectable_label(self.target_idx == Some(i), n).clicked() {
                                    self.target_idx = Some(i);
                                    if let Some(ti) = self.target_idx {
                                        self.goto_target(ti);
                                        self.refresh(ctx);
                                        self.update_wavelength();
                                    }
                                }
                            }
                        });
                    ui.add_space(10.0);
                    ui.label(self.t("GOTO wavelength (Å)"));
                    ui.add_sized([80.0, 22.0], egui::TextEdit::singleline(&mut self.ashift_text));
                    if ui.button(self.t("GOTO")).clicked() {
                        self.goto_wavelength();
                        self.refresh(ctx);
                        self.update_wavelength();
                    }
                    ui.add_space(10.0);
                    ui.label(self.t("Pixel shift"));
                    // Spinbox like the Python Spin(enable_events=True): up/down arrow
                    // buttons nudge the value and refresh the display immediately.
                    let mut v = self.shift;
                    let sresp = ui.add_sized(
                        [56.0, 22.0],
                        egui::DragValue::new(&mut v).speed(1).range(-999..=999),
                    );
                    let mut changed = sresp.changed();
                    let mut nudge = 0i64;
                    ui.scope(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(2.0, 0.0);
                        ui.spacing_mut().button_padding = egui::vec2(2.0, 0.0);
                        ui.vertical(|ui| {
                            if ui.add(egui::Button::new("▲").small()).clicked() {
                                nudge = 1;
                            }
                            if ui.add(egui::Button::new("▼").small()).clicked() {
                                nudge = -1;
                            }
                        });
                    });
                    if nudge != 0 {
                        v = (self.shift + nudge).clamp(-999, 999);
                        changed = true;
                    }
                    if changed {
                        self.shift = v;
                        self.shift_text = v.to_string();
                        if self.loaded.is_some() {
                            self.refresh(ctx);
                            self.update_wavelength();
                        }
                    }
                    ui.add_space(10.0);
                    ui.label(&self.wavelength_text);
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label(self.t("Dispersion (Å/pixel)"));
                    let resp = ui.add_sized([100.0, 22.0], egui::TextEdit::singleline(&mut self.disp_text));
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        self.apply_dispersion();
                        self.update_wavelength();
                    }
                    if ui.button(self.t("Auto dispersion")).clicked() {
                        if self.loaded.is_some() && self.anchor_idx.is_some() {
                            self.auto_dispersion();
                            self.refresh(ctx);
                            self.update_wavelength();
                        } else {
                            self.popup = Some(self.t("First load a file and press start analysis and choose anchor line!"));
                        }
                    }
                    if ui.button(self.t("Apply shift")).clicked() {
                        self.apply_shift(ctx);
                    }
                });
                ui.separator();
                self.draw_canvas(ui);
                ui.label(self.status_text());
                if self.popup.is_some() {
                    let msg = self.popup.clone().unwrap_or_default();
                    let mut clear = false;
                    ui.horizontal(|ui| {
                        ui.colored_label(Color32::from_rgb(255, 90, 90), &msg);
                        if ui.button(self.t("OK")).clicked() {
                            clear = true;
                        }
                    });
                    if clear {
                        self.popup = None;
                    }
                }
            });
            if ctx.input(|i| i.viewport().close_requested()) {
                close_after = true;
            }
        });
        if close_after {
            self.open = false;
            action = Action::Close(Some(self.shift));
        }
        action
    }

    fn apply_dispersion(&mut self) {
        let v = self.disp_text.trim().parse::<f64>();
        match v {
            Ok(d) if d > 0.0 => {
                self.dispersion = Some((d * 1e6).round() / 1e6);
                self.env.dispersion = self.dispersion.unwrap();
            }
            _ => {
                self.popup = Some(self.t("Invalid dispersion"));
            }
        }
    }

    fn apply_shift(&mut self, ctx: &egui::Context) {
        match self.shift_text.trim().parse::<i64>() {
            Ok(v) => {
                self.shift = v;
                self.refresh(ctx);
                self.update_wavelength();
            }
            Err(_) => {
                self.popup = Some(format!("{}: {}: {}", self.t("Error"), self.t("invalid shift"), self.shift_text))
            }
        }
    }

    fn goto_target(&mut self, ti: usize) {
        if self.loaded.is_none() || self.anchor_idx.is_none() || self.dispersion.is_none() {
            self.popup = Some(self.t("Not ready to do that yet: load file and find dispersion first!"));
            return;
        }
        let l = self.loaded.as_ref().unwrap();
        let (ai, d) = (self.anchor_idx.unwrap(), self.dispersion.unwrap());
        let anchor_guess = self.anchors.0[ai];
        let target = self.targets.0[ti];
        let shift = ((target - anchor_guess) / d) as i64;
        let within_all = l.curve.iter().all(|&c| {
            let p = c as f64 + shift as f64;
            p >= 0.0 && p <= l.mean.w as f64
        });
        let within_any = l.curve.iter().any(|&c| {
            let p = c as f64 + shift as f64;
            p >= 0.0 && p <= l.mean.w as f64
        });
        if within_any {
            self.shift = shift;
            self.shift_text = shift.to_string();
            if !within_all {
                self.popup = Some(self.t("Warning: Line is only partially within frame"));
            }
        } else {
            self.popup = Some(self.t("That line does not appear to be in image!"));
        }
    }

    fn goto_wavelength(&mut self) {
        if self.loaded.is_none() || self.anchor_idx.is_none() || self.dispersion.is_none() {
            self.popup = Some(
                self.t("Not ready to GOTO wavelength. First load a file and press start analysis and choose anchor line!"),
            );
            return;
        }
        let l = self.loaded.as_ref().unwrap();
        let (ai, d) = (self.anchor_idx.unwrap(), self.dispersion.unwrap());
        let gotolambda = match self.ashift_text.trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => {
                self.popup = Some(self.t("invalid wavelength"));
                return;
            }
        };
        let anchor_guess = self.anchors.0[ai];
        let shift = ((gotolambda - anchor_guess) / d) as i64;
        let within_any = l.curve.iter().any(|&c| {
            let p = c as f64 + shift as f64;
            p >= 0.0 && p <= l.mean.w as f64
        });
        let within_all = l.curve.iter().all(|&c| {
            let p = c as f64 + shift as f64;
            p >= 0.0 && p <= l.mean.w as f64
        });
        if within_any {
            self.shift = shift;
            self.shift_text = shift.to_string();
            if !within_all {
                self.popup = Some(self.t("Warning: Line is only partially within frame"));
            }
        } else {
            self.popup = Some(self.t("That line does not appear to be in image!"));
        }
    }

    fn update_wavelength(&mut self) {
        if self.loaded.is_none() {
            self.wavelength_text = self.wavelength_none();
            return;
        }
        if let (Some(ai), Some(d)) = (self.anchor_idx, self.dispersion) {
            let anchor_val = self.anchors.0[ai];
            let v = anchor_val + self.shift as f64 * d;
            self.ashift_text = format!("{:.3}", v);
            self.wavelength_text = format!("{}: {:.3}Å", self.t("Wavelength shift"), self.shift as f64 * d);
        } else {
            self.wavelength_text = self.wavelength_none();
        }
    }

    fn draw_canvas(&self, ui: &mut egui::Ui) {
        // Grow/shrink with the native window; leave room for the status line and message.
        let reserve = if self.popup.is_some() { 84.0 } else { 46.0 };
        let h = (ui.available_height() - reserve).max(180.0);
        let size = egui::vec2(ui.available_width(), h);
        let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
        let r = resp.rect;
        let pad = 44.0;
        // Leave room above the axes for their titles, otherwise the labels get clipped.
        let top_pad = 26.0;
        let midx = r.center().x;
        let midy = r.center().y;
        let ax1 = egui::Rect::from_min_max(egui::pos2(r.left() + pad, r.top() + top_pad), egui::pos2(midx - 24.0, midy - 30.0));
        let ax2 = egui::Rect::from_min_max(egui::pos2(midx + pad, r.top() + top_pad), egui::pos2(r.right() - 10.0, midy - 30.0));
        let ax3 = egui::Rect::from_min_max(egui::pos2(r.left() + pad, midy + 10.0), egui::pos2(midx - 24.0, r.bottom() - 10.0));
        let ax4 = egui::Rect::from_min_max(egui::pos2(midx + pad, midy + 10.0), egui::pos2(r.right() - 10.0, r.bottom() - 10.0));
        let ax1_title = format!("{} / {}", self.t("X axis"), self.t("Y axis"));
        let ax2_title = self.t("spectrum");
        for (rect, label) in [(ax1, ax1_title), (ax2, ax2_title), (ax3, String::new()), (ax4, String::new())] {
            painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0_f32, Color32::from_gray(110)));
            painter.text(rect.left_top() + egui::vec2(4.0, -16.0), egui::Align2::LEFT_TOP, label, egui::FontId::proportional(11.0), Color32::from_gray(160));
        }
        let grid = egui::Stroke::new(0.5_f32, Color32::from_gray(55));
        let tick_col = Color32::from_gray(170);
        let tick_font = egui::FontId::proportional(10.0);
        let Some(l) = &self.loaded else { return };
        let iw = l.mean.w as f64;
        let ih = l.mean.h as f64;
        // ax1: mean image + fitted spectral line
        if let Some(t) = &self.mean_tex {
            painter.image(t.id(), ax1, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        }
        let mapx = |v: f64| ax1.left() + (v / (iw - 1.0).max(1.0)) as f32 * ax1.width();
        let mapy = |v: f64| ax1.top() + (v / (ih - 1.0).max(1.0)) as f32 * ax1.height();
        // ax1 grid + tick labels (matplotlib grid() equivalent).
        let (xt, xstep) = nice_ticks(0.0, iw - 1.0, 8);
        for &t in &xt {
            let x = mapx(t);
            painter.line_segment([egui::pos2(x, ax1.top()), egui::pos2(x, ax1.bottom())], grid);
            painter.text(egui::pos2(x, ax1.bottom() + 2.0), egui::Align2::CENTER_TOP, fmt_tick(t, xstep), tick_font.clone(), tick_col);
        }
        let (yt, ystep) = nice_ticks(0.0, ih - 1.0, 8);
        for &t in &yt {
            let y = mapy(t);
            painter.line_segment([egui::pos2(ax1.left(), y), egui::pos2(ax1.right(), y)], grid);
            painter.text(egui::pos2(ax1.left() - 4.0, y), egui::Align2::RIGHT_CENTER, fmt_tick(t, ystep), tick_font.clone(), tick_col);
        }
        painter.text(egui::pos2(ax1.center().x, ax1.bottom() + 16.0), egui::Align2::CENTER_TOP, self.t("X axis"), tick_font.clone(), tick_col);
        let blue_pts: Vec<egui::Pos2> = l.curve.iter().enumerate().map(|(i, &c)| egui::pos2(mapx(c), mapy(i as f64))).collect();
        painter.add(egui::Shape::line(blue_pts, egui::Stroke::new(1.0_f32, Color32::from_rgb(60, 120, 255))));
        let red_pts: Vec<egui::Pos2> = l.curve.iter().enumerate().map(|(i, &c)| egui::pos2(mapx(c + self.shift as f64), mapy(i as f64))).collect();
        draw_dashed(&painter, &red_pts, Color32::from_rgb(255, 40, 40));
        // ax2: spectrum
        let mid = l.mean.h / 2;
        let spectrum2: Vec<f64> = (0..l.mean.w).map(|x| l.mean.g(mid, x) as f64).collect();
        let lspec = log_vec(&spectrum2);
        let anchor_px = l.curve[mid];
        match (self.anchor_idx, self.dispersion) {
            (Some(ai), Some(d)) => {
                let anchor_val = self.anchors.0[ai];
                let lo_clip = -anchor_px as f64 * d + anchor_val;
                let hi_clip = (spectrum2.len() as f64 - anchor_px as f64) * d + anchor_val;
                let (first, step, refy) = &self.alps;
                let refs: Vec<(f64, f64)> = (0..refy.len())
                    .map(|i| (*first + *step * i as f64, refy[i] as f64 / 255.0))
                    .filter(|(x, _)| *x >= lo_clip && *x < hi_clip)
                    .collect();
                let (dlo, dhi) = minmax(&lspec);
                // ax2 grid + tick labels (wavelength on X, log intensity on Y).
                let xmap = |v: f64| ax2.left() + ((v - lo_clip) / (hi_clip - lo_clip).max(1e-9)) as f32 * ax2.width();
                let ymap = |v: f64| ax2.bottom() - ((v - dlo) / (dhi - dlo).max(1e-9)) as f32 * ax2.height();
                let (xt, xstep) = nice_ticks(lo_clip, hi_clip, 8);
                for &t in &xt {
                    let x = xmap(t);
                    painter.line_segment([egui::pos2(x, ax2.top()), egui::pos2(x, ax2.bottom())], grid);
                    painter.text(egui::pos2(x, ax2.bottom() + 2.0), egui::Align2::CENTER_TOP, fmt_tick(t, xstep), tick_font.clone(), tick_col);
                }
                let (yt, ystep) = nice_ticks(dlo, dhi, 5);
                for &t in &yt {
                    let y = ymap(t);
                    painter.line_segment([egui::pos2(ax2.left(), y), egui::pos2(ax2.right(), y)], grid);
                    painter.text(egui::pos2(ax2.left() - 4.0, y), egui::Align2::RIGHT_CENTER, fmt_tick(t, ystep), tick_font.clone(), tick_col);
                }
                if !refs.is_empty() && hi_clip > lo_clip {
                    let (rlo, rhi) = refs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, b| {
                        (a.0.min(b.1), a.1.max(b.1))
                    });
                    let xr = (hi_clip - lo_clip).max(1.0);
                    let pts: Vec<egui::Pos2> = refs.iter().map(|(x, y)| {
                        let px = ax2.left() + ((*x - lo_clip) / xr) as f32 * ax2.width();
                        let py = ax2.bottom() - ((*y - rlo) / (rhi - rlo).max(1e-9)) as f32 * ax2.height();
                        egui::pos2(px, py)
                    }).collect();
                    painter.add(egui::Shape::line(pts, egui::Stroke::new(1.0_f32, Color32::from_rgb(200, 90, 220))));
                    painter.text(ax2.right_top() + egui::vec2(-4.0, 14.0), egui::Align2::RIGHT_TOP, self.t("reference"), tick_font.clone(), Color32::from_rgb(200, 90, 220));
                }
                let pts: Vec<egui::Pos2> = lspec.iter().enumerate().map(|(i, &y)| {
                    let x = (i as f64 - anchor_px as f64) * d + anchor_val;
                    let px = ax2.left() + ((x - lo_clip) / (hi_clip - lo_clip).max(1e-9)) as f32 * ax2.width();
                    let py = ax2.bottom() - ((y - dlo) / (dhi - dlo).max(1e-9)) as f32 * ax2.height();
                    egui::pos2(px, py)
                }).collect();
                painter.add(egui::Shape::line(pts, egui::Stroke::new(1.0_f32, Color32::from_rgb(60, 200, 80))));
                let line_at = |v: f64, col: Color32, dashed: bool| {
                    let x = ax2.left() + ((v - lo_clip) / (hi_clip - lo_clip).max(1e-9)) as f32 * ax2.width();
                    if dashed {
                        draw_dashed(&painter, &[egui::pos2(x, ax2.top()), egui::pos2(x, ax2.bottom())], col);
                    } else {
                        painter.line_segment([egui::pos2(x, ax2.top()), egui::pos2(x, ax2.bottom())], egui::Stroke::new(1.0_f32, col));
                    }
                };
                line_at(anchor_val, Color32::from_rgb(60, 120, 255), false);
                line_at(anchor_val + self.shift as f64 * d, Color32::from_rgb(255, 40, 40), true);
                painter.text(egui::pos2(ax2.center().x, ax2.bottom() + 16.0), egui::Align2::CENTER_TOP,
                    format!("{} {:.4} Å/pixel", self.t("wavelength(Å); dispersion:"), d), tick_font.clone(), tick_col);
                painter.text(ax2.right_top() + egui::vec2(-4.0, 2.0), egui::Align2::RIGHT_TOP, self.t("data"), tick_font.clone(), Color32::from_rgb(60, 200, 80));
            }
            _ => {
                let (lo, hi) = minmax(&lspec);
                let n = lspec.len() as f64;
                // ax2 grid + tick labels (pixel on X, log intensity on Y).
                let (xt, xstep) = nice_ticks(0.0, n - 1.0, 8);
                for &t in &xt {
                    let x = ax2.left() + (t / (n - 1.0).max(1.0)) as f32 * ax2.width();
                    painter.line_segment([egui::pos2(x, ax2.top()), egui::pos2(x, ax2.bottom())], grid);
                    painter.text(egui::pos2(x, ax2.bottom() + 2.0), egui::Align2::CENTER_TOP, fmt_tick(t, xstep), tick_font.clone(), tick_col);
                }
                let (yt, ystep) = nice_ticks(lo, hi, 5);
                for &t in &yt {
                    let y = ax2.bottom() - ((t - lo) / (hi - lo).max(1e-9)) as f32 * ax2.height();
                    painter.line_segment([egui::pos2(ax2.left(), y), egui::pos2(ax2.right(), y)], grid);
                    painter.text(egui::pos2(ax2.left() - 4.0, y), egui::Align2::RIGHT_CENTER, fmt_tick(t, ystep), tick_font.clone(), tick_col);
                }
                painter.text(ax2.right_top() + egui::vec2(-4.0, 2.0), egui::Align2::RIGHT_TOP, self.t("data"), tick_font.clone(), Color32::from_rgb(60, 200, 80));
                let pts: Vec<egui::Pos2> = lspec.iter().enumerate().map(|(i, &y)| {
                    let px = ax2.left() + (i as f64 / (n - 1.0).max(1.0)) as f32 * ax2.width();
                    let py = ax2.bottom() - ((y - lo) / (hi - lo).max(1e-9)) as f32 * ax2.height();
                    egui::pos2(px, py)
                }).collect();
                painter.add(egui::Shape::line(pts, egui::Stroke::new(1.0_f32, Color32::from_rgb(60, 200, 80))));
                let px = ax2.left() + (anchor_px / (n - 1.0).max(1.0)) as f32 * ax2.width();
                painter.line_segment([egui::pos2(px, ax2.top()), egui::pos2(px, ax2.bottom())], egui::Stroke::new(1.0_f32, Color32::from_rgb(60, 120, 255)));
                draw_dashed(&painter, &[egui::pos2(px + self.shift as f32, ax2.top()), egui::pos2(px + self.shift as f32, ax2.bottom())], Color32::from_rgb(255, 40, 40));
            }
        }
        // ax3/ax4: clahe and protus, drawn with the image aspect preserved
        // (matplotlib imshow aspect='equal') so the solar disk stays circular.
        for (rect, tex) in [(ax3, &self.cc_tex), (ax4, &self.protus_tex)] {
            if let Some(t) = tex {
                let isz = t.size_vec2();
                let s = (rect.width() / isz.x.max(1.0)).min(rect.height() / isz.y.max(1.0));
                let img_rect = egui::Rect::from_center_size(rect.center(), isz * s);
                painter.image(t.id(), img_rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
            }
        }
    }
}

fn disk_img(info: &SerInfo, d: Vec<u16>) -> Img {
    Img { h: info.ih, w: info.frame_count, d }
}

fn scaled_circle(c: (f64, f64, f64), f: f64) -> (f64, f64, f64) {
    if c.0 == -1.0 { c } else { ((c.0 * f).trunc(), (c.1 * f).trunc(), (c.2 * f).trunc()) }
}

/// tuple_downscale for the four ellipse borders (Python truncates to int).
fn scaled_borders(b: [f64; 4], f: f64) -> [f64; 4] {
    [(b[0] * f).trunc(), (b[1] * f).trunc(), (b[2] * f).trunc(), (b[3] * f).trunc()]
}

fn scaled_bounds(b: (usize, usize), f: f64) -> (usize, usize) {
    ((b.0 as f64 * f).trunc() as usize, (b.1 as f64 * f).trunc() as usize)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}

/// Matplotlib-style "nice" tick positions covering [lo, hi] (returns ticks and the step).
fn nice_ticks(lo: f64, hi: f64, target: usize) -> (Vec<f64>, f64) {
    if !(hi > lo) {
        return (Vec::new(), 1.0);
    }
    let span = hi - lo;
    let raw = span / target.max(1) as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .iter()
        .map(|m| m * mag)
        .find(|&s| span / s <= target as f64 + 0.5)
        .unwrap_or(10.0 * mag);
    let mut v = Vec::new();
    let mut x = (lo / step).ceil() * step;
    while x <= hi + 1e-9 {
        v.push(x);
        x += step;
    }
    (v, step)
}

fn fmt_tick(v: f64, step: f64) -> String {
    let v = if v == 0.0 { 0.0 } else { v }; // avoid "-0"
    let dec = if step >= 1.0 { 0 } else { (-step.log10()).ceil().max(0.0) as usize };
    format!("{:.*}", dec, v)
}

fn minmax(v: &[f64]) -> (f64, f64) {
    v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &b| (a.0.min(b), a.1.max(b)))
}

fn log_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| if x > 0.0 { x.ln() } else { 0.0 }).collect()
}

/// np.interp(j, xs, ys) with clamping to the end values.
fn interp_linear(j: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if j <= xs[0] { return ys[0]; }
    let n = xs.len();
    if j >= xs[n - 1] { return ys[n - 1]; }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= j { lo = mid; } else { hi = mid; }
    }
    let t = (j - xs[lo]) / (xs[hi] - xs[lo]);
    ys[lo] + t * (ys[hi] - ys[lo])
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 { return 0.0; }
    let ma = a[..n].iter().sum::<f64>() / n as f64;
    let mb = b[..n].iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let x = a[i] - ma;
        let y = b[i] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt()).max(1e-300)
}

/// cv2.resize INTER_LINEAR with fx = fy = f.
fn bilinear_u16(img: &Img, f: f64) -> Img {
    let ow = ((img.w as f64 * f).round() as usize).max(1);
    let oh = ((img.h as f64 * f).round() as usize).max(1);
    let mut d = vec![0u16; ow * oh];
    for y in 0..oh {
        let sy = ((y as f64 + 0.5) / f) - 0.5;
        let y0 = sy.floor();
        let fy = sy - y0;
        let y0 = y0.max(0.0) as usize;
        let y1 = (y0 + 1).min(img.h - 1);
        for x in 0..ow {
            let sx = ((x as f64 + 0.5) / f) - 0.5;
            let x0 = sx.floor();
            let fx = sx - x0;
            let x0 = x0.max(0.0) as usize;
            let x1 = (x0 + 1).min(img.w - 1);
            let v00 = img.g(y0, x0) as f64;
            let v01 = img.g(y0, x1) as f64;
            let v10 = img.g(y1, x0) as f64;
            let v11 = img.g(y1, x1) as f64;
            let top = v00 + fx * (v01 - v00);
            let bot = v10 + fx * (v11 - v10);
            d[y * ow + x] = (top + fy * (bot - top)).round().clamp(0.0, 65535.0) as u16;
        }
    }
    Img { h: oh, w: ow, d }
}

fn draw_dashed(painter: &egui::Painter, pts: &[egui::Pos2], col: Color32) {
    for (i, w) in pts.windows(2).enumerate() {
        if i % 2 == 0 {
            painter.line_segment([w[0], w[1]], egui::Stroke::new(1.0_f32, col));
        }
    }
}
