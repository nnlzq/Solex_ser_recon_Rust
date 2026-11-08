//! Embedded resources (language packs, flags, reference solar spectrum) and
//! SHG_config.txt persistence.

use serde_json::{Map, Value};
use std::path::PathBuf;

pub struct Lang {
    pub name: String,
    pub dict: Map<String, Value>,
    pub flag: &'static [u8],
}

macro_rules! lang_entry {
    ($file:literal, $flag:literal) => {{
        let v: Value = serde_json::from_str(include_str!(concat!("../../../language_data/", $file)))
            .unwrap_or(Value::Null);
        let name = v
            .get("_lang_name")
            .and_then(|x| x.as_str())
            .unwrap_or("English")
            .to_string();
        let dict = match v {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        Lang { name, dict, flag: include_bytes!(concat!("../../../language_data/", $flag)) }
    }};
}

/// Languages in the same (sorted file name) order as UI_handler.read_langs().
pub fn langs() -> Vec<Lang> {
    vec![
        lang_entry!("dict_lang_DEU.txt", "flagDEU.png"),
        lang_entry!("dict_lang_EN.txt", "flagEN.png"),
        lang_entry!("dict_lang_ESP.txt", "flagESP.png"),
        lang_entry!("dict_lang_FR.txt", "flagFR.png"),
        lang_entry!("dict_lang_IT.txt", "flagIT.png"),
        lang_entry!("dict_lang_JPY.txt", "flagJPY.png"),
        lang_entry!("dict_lang_KOR.txt", "flagKOR.png"),
        lang_entry!("dict_lang_MON.txt", "flagMON.png"),
        lang_entry!("dict_lang_PT.txt", "flagPT.png"),
        lang_entry!("dict_lang_SC.txt", "flagSC.png"),
        lang_entry!("dict_lang_TC.txt", "flagTC.png"),
        lang_entry!("dict_lang_UKR.txt", "flagUKR.png"),
    ]
}

/// alps solar reference spectrum: (first, step, y as 0..255).
pub fn alps() -> (f64, f64, Vec<u8>) {
    let raw: &[u8] = include_bytes!("../../assets/alps_spectrum.bin");
    let first = i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as f64;
    let step = f64::from_le_bytes([raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15]]);
    let n = u32::from_le_bytes([raw[16], raw[17], raw[18], raw[19]]) as usize;
    (first, step, raw[20..20 + n].to_vec())
}

fn read_lines(text: &str) -> (Vec<f64>, Vec<String>, Vec<String>) {
    let mut l = Vec::new();
    let mut names = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() { continue; }
        let mut it = line.split_whitespace();
        if let (Some(a), Some(b)) = (it.next(), it.next()) {
            if let Ok(v) = a.parse::<f64>() {
                l.push(v);
                names.push(b.to_string());
            }
        }
    }
    let names_num: Vec<String> = names
        .iter()
        .zip(l.iter())
        .map(|(n, v)| format!("{}({})", n, v))
        .collect();
    (l, names, names_num)
}

pub fn anchor_lines() -> (Vec<f64>, Vec<String>, Vec<String>) {
    read_lines(include_str!("../../../line_data/anchor_candidates.txt"))
}

pub fn target_lines() -> (Vec<f64>, Vec<String>, Vec<String>) {
    read_lines(include_str!("../../../line_data/line_targets.txt"))
}

pub fn config_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    Some(dir.join("SHG_config.txt"))
}

pub fn load_config() -> Map<String, Value> {
    let p = match config_path() { Some(p) => p, None => return Map::new() };
    let text = match std::fs::read_to_string(&p) { Ok(t) => t, Err(_) => return Map::new() };
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    }
}

pub fn save_config(cfg: &Value) {
    if let Some(p) = config_path() {
        if let Ok(text) = serde_json::to_string_pretty(cfg) {
            let _ = std::fs::write(p, text);
        }
    }
}

pub fn cfg_bool(cfg: &Map<String, Value>, key: &str, default: bool) -> bool {
    cfg.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

pub fn cfg_str(cfg: &Map<String, Value>, key: &str, default: &str) -> String {
    cfg.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| default.to_string())
}
