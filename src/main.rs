#![allow(clippy::collapsible_else_if)]
#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use csv::{ReaderBuilder, StringRecord};
use eframe::egui::{self, RichText, Ui};
use egui::{CornerRadius, Frame, UiBuilder};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

const EXTREMELY_LARGE_FILE_SIZE: u64 = 50_000_000_000;
const LARGE_FILE_CACHE_SIZE: usize = 500;
const COLLAPSED_HEIGHT: f32 = 24.0;
const EXPANDED_ROW_MIN_HEIGHT: f32 = 100.0;
const EXPANDED_JSON_HEIGHT: f32 = 400.0;
const EXPANDED_TABLE_ROW_HEIGHT: f32 = 20.0;
const BUFFER_HEIGHT: f32 = 5000.0;
const CACHE_SIZE: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileFormat {
    JSONL,
    CSV,
    TSV,
}

impl FileFormat {
    fn from_extension(path: &str) -> Self {
        let path_lower = path.to_lowercase();
        if path_lower.ends_with(".csv") {
            FileFormat::CSV
        } else if path_lower.ends_with(".tsv") {
            FileFormat::TSV
        } else {
            FileFormat::JSONL
        }
    }

    fn display_name(&self) -> &'static str {
        match self {
            FileFormat::JSONL => "JSONL",
            FileFormat::CSV => "CSV",
            FileFormat::TSV => "TSV",
        }
    }

    fn delimiter(&self) -> Option<u8> {
        match self {
            FileFormat::CSV => Some(b','),
            FileFormat::TSV => Some(b'\t'),
            FileFormat::JSONL => None,
        }
    }
}

#[derive(Debug, Default)]
struct SearchState {
    term: String,
    matches: HashSet<usize>,
    current_match_idx: Option<usize>,
    match_indices: Vec<usize>,
    count: usize,
    in_progress: bool,
    case_sensitive: bool,
}

struct CacheEntry {
    record: Vec<String>,
    last_access: Instant,
}

trait FormatReader: Send + Sync {
    fn total_records(&self) -> usize;
    fn get_record(&self, index: usize) -> Option<String>;
    fn get_parsed_record(&self, index: usize) -> Option<Vec<String>>;
    fn is_match(&self, index: usize, term: &str, case_sensitive: bool) -> bool;
    fn has_header(&self) -> bool;
    fn get_header(&self) -> Option<Vec<String>>;
    fn format(&self) -> FileFormat;
    fn preload_around(&self, index: usize);
    fn estimate_parsed_height(&self, index: usize, text_wrapping: bool, max_width: f32) -> f32;
}

struct UnifiedReader {
    mmap: Arc<Mmap>,
    offsets: Arc<Vec<usize>>,
    file_format: FileFormat,
    header: Option<Vec<String>>,

    // Cache for parsed records
    record_cache: Arc<Mutex<HashMap<usize, CacheEntry>>>,
    cache_access_queue: Arc<Mutex<VecDeque<usize>>>,

    file_size: u64,
}

impl UnifiedReader {
    fn new(file: &File, file_path: &str, format: FileFormat) -> io::Result<Self> {
        let metadata = file.metadata()?;
        let file_size = metadata.len();

        // Memory map the file for efficient access
        let mmap = unsafe { Mmap::map(file)? };
        let mmap_arc = Arc::new(mmap);

        // Build line offsets index - use format-specific method for CSV/TSV
        let offsets = match format {
            FileFormat::CSV | FileFormat::TSV => {
                let delimiter = format.delimiter().unwrap_or(b',');
                Arc::new(Self::build_csv_offsets(&mmap_arc, delimiter))
            }
            _ => Arc::new(Self::build_offsets(&mmap_arc)),
        };

        // Initialize header for CSV/TSV files
        let header = match format {
            FileFormat::CSV | FileFormat::TSV => {
                let delimiter = format.delimiter().unwrap_or(b',');
                Self::detect_header(file_path, delimiter)?
            }
            _ => None,
        };

        Ok(Self {
            mmap: mmap_arc,
            offsets,
            file_format: format,
            header,
            record_cache: Arc::new(Mutex::new(HashMap::new())),
            cache_access_queue: Arc::new(Mutex::new(VecDeque::new())),
            file_size,
        })
    }

    // Add this new method to the UnifiedReader implementation
    fn build_csv_offsets(mmap: &Mmap, delimiter: u8) -> Vec<usize> {
        let mut offsets = vec![0];
        let mut in_quotes = false;
        let quote_char = b'"'; // Standard quote character for CSV

        const CHUNK_SIZE: usize = 1024 * 1024; // Process in 1MB chunks

        let mut chunk_start = 0;
        while chunk_start < mmap.len() {
            let chunk_end = std::cmp::min(chunk_start + CHUNK_SIZE, mmap.len());
            let chunk = &mmap[chunk_start..chunk_end];

            let mut pos = 0;
            while pos < chunk.len() {
                let c = chunk[pos];

                // Handle quotes - toggle quote state if not escaped
                if c == quote_char {
                    // Check for escaped quotes (two quotes in a row)
                    if in_quotes && pos + 1 < chunk.len() && chunk[pos + 1] == quote_char {
                        // Skip the second quote (escaped quote)
                        pos += 1;
                    } else {
                        // Toggle quote state
                        in_quotes = !in_quotes;
                    }
                }
                // Only treat newlines as record separators when not in quotes
                else if (c == b'\n'
                    || (c == b'\r' && pos + 1 < chunk.len() && chunk[pos + 1] == b'\n'))
                    && !in_quotes
                {
                    let nl_adjust =
                        if c == b'\r' && pos + 1 < chunk.len() && chunk[pos + 1] == b'\n' {
                            // For CRLF, skip both characters
                            pos += 1;
                            2
                        } else {
                            // For LF only
                            1
                        };

                    let abs_pos = chunk_start + pos + nl_adjust;
                    if abs_pos <= mmap.len() {
                        offsets.push(abs_pos);
                    }
                }

                pos += 1;
            }

            // Note: in_quotes state persists across chunks automatically
            chunk_start = chunk_end;
        }

        // Safety check: if we end the file still in quotes, we might have malformed CSV
        // In this case, we'll just add the end of file as the final offset
        if offsets.last().map_or(true, |&off| off < mmap.len()) {
            offsets.push(mmap.len());
        }

        // Remove duplicate at the end if present
        if offsets.len() >= 2 && offsets[offsets.len() - 1] == offsets[offsets.len() - 2] {
            offsets.pop();
        }

        offsets
    }

    fn build_offsets(mmap: &Mmap) -> Vec<usize> {
        let mut offsets = vec![0];

        const CHUNK_SIZE: usize = 1024 * 1024; // Process in 1MB chunks

        let mut chunk_start = 0;
        while chunk_start < mmap.len() {
            let chunk_end = std::cmp::min(chunk_start + CHUNK_SIZE, mmap.len());
            let chunk = &mmap[chunk_start..chunk_end];

            let mut newline_pos = 0;
            while let Some(nl_pos) = memchr::memchr(b'\n', &chunk[newline_pos..]) {
                let abs_pos = chunk_start + newline_pos + nl_pos + 1;

                if abs_pos <= mmap.len() {
                    offsets.push(abs_pos);
                }
                newline_pos += nl_pos + 1;
            }

            chunk_start = chunk_end;
        }

        // Ensure the last offset is included if needed
        if offsets.last().map_or(true, |&off| off < mmap.len()) {
            offsets.push(mmap.len());
        }

        // Remove duplicate at the end if present
        if offsets.len() >= 2 && offsets[offsets.len() - 1] == offsets[offsets.len() - 2] {
            offsets.pop();
        }

        offsets
    }

    fn detect_header(file_path: &str, delimiter: u8) -> io::Result<Option<Vec<String>>> {
        // Try to read headers using CSV reader
        let file = File::open(file_path)?;

        let mut reader = ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(true)
            .quoting(true)
            .double_quote(true)
            .escape(Some(b'\\'))
            .flexible(true)
            .from_reader(BufReader::new(file));

        match reader.headers() {
            Ok(headers) => Ok(Some(headers.iter().map(|s| s.to_string()).collect())),
            Err(_) => {
                // Fallback: read first line and check if it contains delimiters
                let file = File::open(file_path)?;
                let mut buf_reader = BufReader::new(file);
                let mut first_line = String::new();
                if buf_reader.read_line(&mut first_line).is_ok() && !first_line.is_empty() {
                    // Just return None as we can't properly parse headers
                    Ok(None)
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn get_raw_line(&self, index: usize) -> Option<String> {
        if index + 1 < self.offsets.len() {
            let start = self.offsets[index];
            let end = self.offsets[index + 1];

            // Trim trailing newlines
            let mut actual_end = end;
            while actual_end > start
                && (self.mmap[actual_end - 1] == b'\n' || self.mmap[actual_end - 1] == b'\r')
            {
                actual_end -= 1;
            }

            if start <= actual_end {
                if let Ok(line) = std::str::from_utf8(&self.mmap[start..actual_end]) {
                    return Some(line.to_string());
                }
            } else if start == end {
                return Some("".to_string());
            }
        }
        None
    }

    fn parse_csv_line(&self, line: &str, delimiter: u8) -> Option<Vec<String>> {
        let mut reader = ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(false)
            .quoting(true)
            .double_quote(true)
            .escape(Some(b'\\'))
            .flexible(true)
            .from_reader(line.as_bytes());

        let mut record = StringRecord::new();
        if reader.read_record(&mut record).ok()? {
            Some(record.iter().map(String::from).collect())
        } else {
            None
        }
    }

    fn get_parsed_record_internal(&self, index: usize) -> Option<Vec<String>> {
        // Check cache first
        {
            let mut cache = self.record_cache.lock().unwrap();
            if let Some(entry) = cache.get_mut(&index) {
                entry.last_access = Instant::now();

                // Update LRU queue
                let mut access_queue = self.cache_access_queue.lock().unwrap();
                access_queue.retain(|&x| x != index);
                access_queue.push_back(index);

                return Some(entry.record.clone());
            }
        }

        // Cache miss - load and parse the record
        let raw_line = self.get_raw_line(index)?;

        let parsed_record = match self.file_format {
            FileFormat::JSONL => vec![raw_line.clone()],
            FileFormat::CSV | FileFormat::TSV => {
                let delimiter = self.file_format.delimiter().unwrap_or(b',');
                match self.parse_csv_line(&raw_line, delimiter) {
                    Some(fields) => fields,
                    None => vec![raw_line.clone()], // Fallback if parsing fails
                }
            }
        };

        // Add to cache
        {
            let mut cache = self.record_cache.lock().unwrap();
            let mut access_queue = self.cache_access_queue.lock().unwrap();

            cache.insert(
                index,
                CacheEntry {
                    record: parsed_record.clone(),
                    last_access: Instant::now(),
                },
            );

            access_queue.retain(|&x| x != index);
            access_queue.push_back(index);

            // Prune cache if necessary
            self.prune_cache(&mut cache, &mut access_queue);
        }

        Some(parsed_record)
    }

    fn prune_cache(&self, cache: &mut HashMap<usize, CacheEntry>, queue: &mut VecDeque<usize>) {
        let effective_cache_size = if self.file_size > EXTREMELY_LARGE_FILE_SIZE {
            LARGE_FILE_CACHE_SIZE
        } else {
            CACHE_SIZE
        };

        if cache.len() <= effective_cache_size {
            return;
        }

        // Ensure queue consistency
        let current_keys: HashSet<usize> = cache.keys().copied().collect();
        queue.retain(|&idx| current_keys.contains(&idx));

        for key in current_keys {
            if !queue.contains(&key) {
                queue.push_back(key);
            }
        }

        // Remove oldest entries
        while cache.len() > effective_cache_size {
            if let Some(idx_to_remove) = queue.pop_front() {
                if cache.contains_key(&idx_to_remove) {
                    cache.remove(&idx_to_remove);
                }
            } else {
                // Inconsistency recovery
                if let Some(key_to_remove) = cache.keys().next().cloned() {
                    cache.remove(&key_to_remove);
                } else {
                    break;
                }
            }
        }
    }

    fn is_record_match(&self, index: usize, term: &str, case_sensitive: bool) -> bool {
        if let Some(record) = self.get_parsed_record_internal(index) {
            // For CSV/TSV check each field
            let search_term = if case_sensitive {
                term.to_string()
            } else {
                term.to_lowercase()
            };

            for field in record {
                let field_to_check = if case_sensitive {
                    field.clone()
                } else {
                    field.to_lowercase()
                };
                if field_to_check.contains(&search_term) {
                    return true;
                }
            }
        }
        false
    }

    fn estimate_parsed_height(&self, index: usize, text_wrapping: bool, max_width: f32) -> f32 {
        if let Some(record) = self.get_parsed_record_internal(index) {
            let base_height = 30.0;
            let row_height = EXPANDED_TABLE_ROW_HEIGHT;

            match self.file_format {
                FileFormat::JSONL => EXPANDED_JSON_HEIGHT,
                FileFormat::CSV | FileFormat::TSV => {
                    if text_wrapping {
                        // Estimate lines needed based on width
                        let approx_chars_per_line = (max_width / 8.0).max(10.0) as usize;
                        let total_lines: usize = record
                            .iter()
                            .map(|field| {
                                // Calculate lines needed for this field
                                let field_len = field.chars().count();
                                // Add 1 for the field name row, then lines for value
                                1 + (field_len + approx_chars_per_line - 1) / approx_chars_per_line
                            })
                            .sum();

                        (base_height + total_lines as f32 * row_height).max(EXPANDED_ROW_MIN_HEIGHT)
                    } else {
                        // Height based on number of fields
                        (base_height + record.len() as f32 * row_height)
                            .max(EXPANDED_ROW_MIN_HEIGHT)
                    }
                }
            }
        } else {
            EXPANDED_ROW_MIN_HEIGHT
        }
    }
}

impl FormatReader for UnifiedReader {
    fn total_records(&self) -> usize {
        if self.offsets.len() < 2 {
            0
        } else {
            self.offsets.len() - 1
        }
    }

    fn get_record(&self, index: usize) -> Option<String> {
        self.get_raw_line(index)
    }

    fn get_parsed_record(&self, index: usize) -> Option<Vec<String>> {
        self.get_parsed_record_internal(index)
    }

    fn is_match(&self, index: usize, term: &str, case_sensitive: bool) -> bool {
        self.is_record_match(index, term, case_sensitive)
    }

    fn has_header(&self) -> bool {
        self.header.is_some()
    }

    fn get_header(&self) -> Option<Vec<String>> {
        self.header.clone()
    }

    fn format(&self) -> FileFormat {
        self.file_format
    }

    fn preload_around(&self, _index: usize) {
        // No-op in this implementation
    }

    fn estimate_parsed_height(&self, index: usize, text_wrapping: bool, max_width: f32) -> f32 {
        self.estimate_parsed_height(index, text_wrapping, max_width)
    }
}

struct JsonlViewer {
    format_reader: Option<Arc<dyn FormatReader>>,
    format_reader_receiver: Option<mpsc::Receiver<Result<Arc<dyn FormatReader>, io::Error>>>,

    search: SearchState,
    search_receiver: Option<mpsc::Receiver<HashSet<usize>>>,
    show_search: bool,
    scroll_to_search: Option<usize>,

    expanded_rows: HashSet<usize>,

    display_cache: HashMap<usize, String>,

    visible_range: (usize, usize),
    total_rows: usize,
    viewport_height: f32,
    dark_mode: bool,

    last_scroll_offset: f32,
    previous_height_cache: HashMap<usize, f32>,
    scroll_memory: Option<f32>,
    stabilize_counter: i32,

    show_message: Option<(String, f32)>,
    message_timer: Option<f64>,

    copy_message: Option<(String, f64)>,

    file_format: FileFormat,
    file_size: u64,
    file_path: Option<String>,
    loading_progress: f32,

    text_wrapping: bool,
    max_column_width: f32,

    last_render_time: Instant,
    render_duration: Duration,
}

impl JsonlViewer {
    fn new() -> Self {
        Self {
            format_reader: None,
            format_reader_receiver: None,

            search: SearchState::default(),
            search_receiver: None,
            show_search: false,
            scroll_to_search: None,

            expanded_rows: HashSet::new(),
            display_cache: HashMap::new(),

            visible_range: (0, 0),
            total_rows: 0,
            viewport_height: 0.0,
            dark_mode: true,

            last_scroll_offset: 0.0,
            previous_height_cache: HashMap::new(),
            scroll_memory: None,
            stabilize_counter: 0,

            show_message: None,
            message_timer: None,

            copy_message: None,

            file_format: FileFormat::JSONL,

            file_size: 0,
            file_path: None,
            loading_progress: 0.0,

            text_wrapping: true,
            max_column_width: 300.0,

            last_render_time: Instant::now(),
            render_duration: Duration::from_millis(0),
        }
    }

    fn open_file(&mut self, path: &str) -> io::Result<()> {
        let file = File::open(path)?;

        let metadata = file.metadata()?;
        self.file_size = metadata.len();
        self.file_path = Some(path.to_string());

        self.format_reader = None;
        self.format_reader_receiver = None;
        self.search = SearchState::default();
        self.search_receiver = None;
        self.expanded_rows.clear();
        self.display_cache.clear();
        self.visible_range = (0, 0);
        self.total_rows = 0;
        self.last_scroll_offset = 0.0;
        self.previous_height_cache.clear();
        self.scroll_memory = None;
        self.stabilize_counter = 0;
        self.loading_progress = 0.0;

        let detected_format = FileFormat::from_extension(path);
        self.file_format = detected_format;

        self.show_message = Some((
            format!("Detected: {}. Loading...", detected_format.display_name()),
            3.0,
        ));
        self.message_timer = None;

        let (sender, receiver) = mpsc::channel();
        self.format_reader_receiver = Some(receiver);

        let path_clone = path.to_string();
        let format = self.file_format;

        thread::spawn(move || {
            let reader_result = match File::open(&path_clone) {
                Ok(file) => UnifiedReader::new(&file, &path_clone, format)
                    .map(|r| Arc::new(r) as Arc<dyn FormatReader>),
                Err(e) => Err(e),
            };

            let _ = sender.send(reader_result);
        });

        Ok(())
    }

    fn start_search(&mut self) {
        if self.search.term.is_empty() || self.format_reader.is_none() {
            self.search.matches.clear();
            self.search.match_indices.clear();
            self.search.current_match_idx = None;
            self.search.count = 0;
            self.search.in_progress = false;
            return;
        }

        self.search.in_progress = true;
        self.search.matches.clear();
        self.search.match_indices.clear();
        self.search.current_match_idx = None;
        self.search.count = 0;
        self.scroll_to_search = None;

        if let Some(reader) = self.format_reader.as_ref().cloned() {
            let search_term = self.search.term.clone();
            let case_sensitive = self.search.case_sensitive;
            let total_records = reader.total_records();

            let search_limit = if self.file_size > 10_000_000_000 {
                (total_records as u64).min(50_000)
            } else if self.file_size > 1_000_000_000 {
                (total_records as u64).min(200_000)
            } else {
                total_records as u64
            };

            let (sender, receiver) = mpsc::channel();
            self.search_receiver = Some(receiver);

            thread::spawn(move || {
                let mut matches = HashSet::new();
                let _start_time = Instant::now();
                let mut last_send_time = Instant::now();

                for row in 0..(search_limit as usize) {
                    if reader.is_match(row, &search_term, case_sensitive) {
                        matches.insert(row);
                    }

                    if row % 5000 == 0 && last_send_time.elapsed() > Duration::from_millis(100) {
                        if sender.send(matches.clone()).is_err() {
                            break;
                        }
                        last_send_time = Instant::now();
                    }

                    if row % 10000 == 0 {
                        thread::sleep(Duration::from_millis(1));
                    }
                }

                let _ = sender.send(matches);
            });
        } else {
            self.search.in_progress = false;
        }
    }

    fn process_search_results(&mut self) {
        if let Some(receiver) = &self.search_receiver {
            let _received_final = false;
            let mut latest_matches: Option<HashSet<usize>> = None;

            while let Ok(matches_update) = receiver.try_recv() {
                latest_matches = Some(matches_update);
            }

            if let Some(matches) = latest_matches {
                self.search.matches = matches;
                self.search.match_indices = self.search.matches.iter().copied().collect();
                self.search.match_indices.sort_unstable();
                self.search.count = self.search.match_indices.len();
            }

            if let Err(mpsc::TryRecvError::Disconnected) = receiver.try_recv() {
                self.search.in_progress = false;
                self.search_receiver = None;

                if !self.search.match_indices.is_empty() && self.search.current_match_idx.is_none()
                {
                    self.search.current_match_idx = Some(0);
                    let first_match_row = self.search.match_indices[0];
                    self.show_message = Some((
                        format!(
                            "Search finished. {} matches. First at line {}.",
                            self.search.count,
                            first_match_row + 1
                        ),
                        3.0,
                    ));
                    self.message_timer = None;
                    self.scroll_to_search = Some(first_match_row);
                    self.previous_height_cache.clear();
                } else if self.search.match_indices.is_empty() {
                    self.show_message =
                        Some(("Search finished. No matches found.".to_string(), 3.0));
                    self.message_timer = None;
                }
            }
        }
    }

    fn next_match(&mut self) {
        if self.search.match_indices.is_empty() {
            return;
        }

        let current = self
            .search
            .current_match_idx
            .unwrap_or(self.search.match_indices.len() - 1);
        let next = (current + 1) % self.search.match_indices.len();

        self.search.current_match_idx = Some(next);
        let target_row = self.search.match_indices[next];

        self.show_message = Some((
            format!(
                "Match {}/{} (Line {})",
                next + 1,
                self.search.count,
                target_row + 1
            ),
            2.0,
        ));
        self.message_timer = None;
        self.scroll_to_search = Some(target_row);
        self.previous_height_cache.clear();
    }

    fn prev_match(&mut self) {
        if self.search.match_indices.is_empty() {
            return;
        }

        let current = self.search.current_match_idx.unwrap_or(0);
        let prev = if current == 0 {
            self.search.match_indices.len() - 1
        } else {
            current - 1
        };

        self.search.current_match_idx = Some(prev);
        let target_row = self.search.match_indices[prev];

        self.show_message = Some((
            format!(
                "Match {}/{} (Line {})",
                prev + 1,
                self.search.count,
                target_row + 1
            ),
            2.0,
        ));
        self.message_timer = None;
        self.scroll_to_search = Some(target_row);
        self.previous_height_cache.clear();
    }

    fn copy_row_content(&mut self, row: usize, ctx: &egui::Context) {
        if let Some(reader) = &self.format_reader {
            if let Some(content) = reader.get_record(row) {
                ctx.copy_text(content);
                self.copy_message =
                    Some((format!("Row {} copied", row + 1), ctx.input(|i| i.time)));
            }
        }
    }

    fn copy_formatted_content(&mut self, row: usize, ctx: &egui::Context) {
        self.ensure_display_cached(row);
        if let Some(formatted) = self.display_cache.get(&row) {
            ctx.copy_text(formatted.clone());
            self.copy_message = Some((
                format!("Formatted row {} copied", row + 1),
                ctx.input(|i| i.time),
            ));
        } else {
            self.copy_row_content(row, ctx);
            self.copy_message = Some((
                format!("Formatted unavailable, copied raw row {}", row + 1),
                ctx.input(|i| i.time),
            ));
        }
    }

    fn row_height(&self, row: usize) -> f32 {
        if self.expanded_rows.contains(&row) {
            match self.file_format {
                FileFormat::JSONL => EXPANDED_JSON_HEIGHT,
                FileFormat::CSV | FileFormat::TSV => {
                    self.format_reader
                        .as_ref()
                        .map_or(EXPANDED_ROW_MIN_HEIGHT, |reader| {
                            reader.estimate_parsed_height(
                                row,
                                self.text_wrapping,
                                self.max_column_width,
                            )
                        })
                }
            }
            .max(EXPANDED_ROW_MIN_HEIGHT)
        } else {
            COLLAPSED_HEIGHT
        }
    }

    fn height_before_row(&self, row: usize) -> f32 {
        if row == 0 {
            return 0.0;
        }

        if let Some(&cached_height) = self.previous_height_cache.get(&(row - 1)) {
            return cached_height + self.row_height(row - 1) + 2.0;
        }

        let mut best_cached_row_idx = 0;
        let mut starting_height = 0.0;
        let mut found_cache = false;

        if let Some(&max_cached_row_before) = self
            .previous_height_cache
            .keys()
            .filter(|&&k| k < row)
            .max()
        {
            starting_height = self.previous_height_cache[&max_cached_row_before];
            best_cached_row_idx = max_cached_row_before;
            found_cache = true;
        }

        let mut current_height = starting_height;
        let start_iter = if found_cache { best_cached_row_idx } else { 0 };

        for i in start_iter..row {
            current_height += self.row_height(i) + 2.0;
        }

        current_height
    }

    fn update_height_cache(&mut self, start_row: usize, end_row: usize) {
        if self.total_rows == 0 {
            return;
        }

        let max_cache_size = if self.file_size > 10_000_000_000 {
            1000
        } else if self.file_size > 1_000_000_000 {
            5000
        } else {
            20000
        };

        if self.previous_height_cache.len() > max_cache_size {
            let keys: Vec<usize> = self.previous_height_cache.keys().copied().collect();
            let center = (start_row + end_row) / 2;
            let mut sorted_keys = keys;
            sorted_keys.sort_by_key(|k| k.abs_diff(center));
            let keys_to_keep: HashSet<usize> = sorted_keys
                .iter()
                .take(max_cache_size / 2)
                .copied()
                .collect();
            self.previous_height_cache
                .retain(|k, _| keys_to_keep.contains(k));
        }

        let mut current_height = self.height_before_row(start_row);

        let effective_end_row = std::cmp::min(end_row + 50, self.total_rows);

        for row in start_row..effective_end_row {
            if self.previous_height_cache.get(&row) != Some(&current_height) {
                self.previous_height_cache.insert(row, current_height);
            }

            current_height += self.row_height(row) + 2.0;

            if self.previous_height_cache.len() >= max_cache_size + 500 {
                break;
            }
        }
    }

    fn row_at_height(&self, target_height: f32) -> usize {
        if self.total_rows == 0 {
            return 0;
        }

        let mut low = 0;
        let mut high = self.total_rows;

        while low < high {
            let mid = low + (high - low) / 2;
            let height_at_mid = self.height_before_row(mid);

            if height_at_mid < target_height {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        if low > 0 && self.height_before_row(low) > target_height {
            low -= 1;
        }

        low.min(self.total_rows.saturating_sub(1))
    }

    fn toggle_row_expansion(&mut self, row: usize, ui: &egui::Ui) {
        let scroll_offset = ui.clip_rect().top() - ui.min_rect().top();

        let height_before = self.height_before_row(row);
        let height_change: f32;

        if self.expanded_rows.contains(&row) {
            let current_expanded_height = self.row_height(row);
            self.expanded_rows.remove(&row);

            height_change = COLLAPSED_HEIGHT - current_expanded_height;
        } else {
            self.expanded_rows.insert(row);
            self.ensure_display_cached(row);
            let new_expanded_height = self.row_height(row);
            height_change = new_expanded_height - COLLAPSED_HEIGHT;
        }

        if height_before < scroll_offset {
            let new_scroll_offset = (scroll_offset + height_change).max(0.0);
            self.scroll_memory = Some(new_scroll_offset);
            self.stabilize_counter = 3;
        }

        self.previous_height_cache.retain(|&k, _| k <= row);
    }

    fn render_tabular_view(&self, ui: &mut Ui, row: usize) {
        let max_fields_to_show = if self.file_size > EXTREMELY_LARGE_FILE_SIZE {
            20
        } else {
            usize::MAX
        };

        if let Some(reader) = &self.format_reader {
            if let Some(fields) = reader.get_parsed_record(row) {
                if fields.is_empty() {
                    ui.label(RichText::new("Row is empty").monospace());
                    return;
                }

                let headers = reader.get_header();
                let _num_columns = fields.len();

                egui::Grid::new(format!("row_{}_grid", row))
                    .striped(true)
                    .num_columns(2)
                    .spacing([10.0, 4.0])
                    .min_col_width(100.0)
                    .show(ui, |ui| {
                        ui.label(RichText::new("Field").strong());
                        ui.label(RichText::new("Value").strong());
                        ui.end_row();

                        for (i, field_value) in fields.iter().take(max_fields_to_show).enumerate() {
                            let field_name = headers
                                .as_ref()
                                .and_then(|h| h.get(i))
                                .map_or_else(|| format!("Field {}", i + 1), |name| name.clone());

                            ui.label(RichText::new(&field_name).monospace().color(
                                if self.dark_mode {
                                    egui::Color32::from_rgb(180, 180, 180)
                                } else {
                                    egui::Color32::from_rgb(70, 70, 70)
                                },
                            ));

                            let text_color = if self.dark_mode {
                                egui::Color32::LIGHT_GRAY
                            } else {
                                egui::Color32::DARK_GRAY
                            };
                            if self.text_wrapping {
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(field_value).monospace().color(text_color),
                                    )
                                    .wrap(),
                                );
                            } else {
                                ui.label(RichText::new(field_value).monospace().color(text_color));
                            }
                            ui.end_row();
                        }

                        if fields.len() > max_fields_to_show {
                            ui.label(RichText::new("...").monospace());
                            ui.label(
                                RichText::new(format!(
                                    "({} more fields)",
                                    fields.len() - max_fields_to_show
                                ))
                                .monospace(),
                            );
                            ui.end_row();
                        }
                    });
                return;
            }
        }

        ui.label(RichText::new("Unable to load/parse row data").monospace());
    }

    fn render_row(&mut self, ui: &mut Ui, row: usize, ctx: &egui::Context) {
        if row >= self.total_rows {
            return;
        }

        if let Some(reader) = &self.format_reader {
            reader.preload_around(row);
        }

        let is_expanded = self.expanded_rows.contains(&row);
        let is_match = self.search.matches.contains(&row);
        let is_current_match = self.search.current_match_idx.map_or(false, |idx| {
            self.search.match_indices.get(idx) == Some(&row)
        });

        let bg_color = if is_current_match {
            if self.dark_mode {
                egui::Color32::from_rgb(80, 55, 0)
            } else {
                egui::Color32::from_rgb(255, 245, 200)
            }
        } else if row % 2 == 0 {
            if self.dark_mode {
                egui::Color32::from_rgb(32, 33, 36)
            } else {
                egui::Color32::from_rgb(245, 245, 245)
            }
        } else {
            if self.dark_mode {
                egui::Color32::from_rgb(40, 41, 45)
            } else {
                egui::Color32::from_rgb(255, 255, 255)
            }
        };

        let border_stroke = if is_current_match {
            egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 165, 0))
        } else if is_match {
            egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 150, 0))
        } else {
            egui::Stroke::NONE
        };

        let frame = Frame::NONE.inner_margin(egui::Margin::symmetric(4, 2));

        let outer_height = self.row_height(row);

        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), outer_height),
            egui::Sense::click(),
        );
        if ui.is_rect_visible(rect) {
            ui.painter().rect(
                rect,
                CornerRadius::ZERO,
                bg_color,
                border_stroke,
                egui::StrokeKind::Outside,
            );

            let mut child_ui = ui.new_child(UiBuilder::new().max_rect(rect));

            frame.show(&mut child_ui, |ui| {
                ui.horizontal(|ui| {
                    let button_text = if is_expanded { "▼" } else { "►" };

                    let expand_response = ui.add(egui::Button::new(button_text).frame(false));
                    if expand_response.clicked() {
                        self.toggle_row_expansion(row, ui);
                    }

                    let row_label_text = format!("[{}]", row + 1);
                    let row_label_style = RichText::new(row_label_text).monospace().strong().color(
                        if is_current_match {
                            egui::Color32::from_rgb(255, 200, 0)
                        } else if is_match {
                            egui::Color32::from_rgb(220, 180, 0)
                        } else if self.dark_mode {
                            egui::Color32::from_rgb(180, 180, 180)
                        } else {
                            egui::Color32::from_rgb(70, 70, 70)
                        },
                    );
                    ui.label(row_label_style);

                    if ui
                        .button("📋")
                        .on_hover_text("Copy formatted/raw content")
                        .clicked()
                    {
                        if is_expanded {
                            self.copy_formatted_content(row, ctx);
                        } else {
                            self.copy_row_content(row, ctx);
                        }
                    }

                    if !is_expanded {
                        if let Some(reader) = &self.format_reader {
                            if let Some(line) = reader.get_record(row) {
                                let preview = Self::create_preview(&line, reader.format(), || {
                                    reader.get_parsed_record(row)
                                });
                                if is_match && !self.search.term.is_empty() {
                                    self.render_highlighted_text(ui, &preview, &self.search.term);
                                } else {
                                    let mut prev_mut = preview;
                                    ui.add(
                                        egui::TextEdit::singleline(&mut prev_mut)
                                            .font(egui::TextStyle::Monospace)
                                            .desired_width(ui.available_width() - 10.0)
                                            .interactive(false)
                                            .frame(false),
                                    );
                                }
                            } else {
                                ui.label(RichText::new("Loading...").monospace().weak());
                            }
                        } else {
                            ui.label(RichText::new("Reader not available").monospace().weak());
                        }
                    }
                });

                if is_expanded {
                    ui.add_space(4.0);
                    self.ensure_display_cached(row);

                    match self.file_format {
                        FileFormat::JSONL => {
                            if let Some(pretty_json) = self.display_cache.get(&row) {
                                egui::ScrollArea::vertical()
                                    .max_height(outer_height - COLLAPSED_HEIGHT.max(25.0))
                                    .id_salt(format!("json_scroll_{}", row))
                                    .show(ui, |ui| {
                                        ui.add(
                                            egui::TextEdit::multiline(&mut pretty_json.clone())
                                                .font(egui::TextStyle::Monospace)
                                                .text_color(if self.dark_mode {
                                                    egui::Color32::LIGHT_GRAY
                                                } else {
                                                    egui::Color32::DARK_GRAY
                                                })
                                                .desired_width(f32::INFINITY)
                                                .interactive(false)
                                                .frame(false),
                                        );
                                    });
                            } else {
                                ui.spinner();
                            }
                        }
                        FileFormat::CSV | FileFormat::TSV => {
                            egui::ScrollArea::vertical()
                                .max_height(outer_height - COLLAPSED_HEIGHT.max(25.0))
                                .id_salt(format!("table_scroll_{}", row))
                                .show(ui, |ui| {
                                    self.render_tabular_view(ui, row);
                                });
                        }
                    }
                }
            });
        }

        if response.clicked() && !response.hovered() {
            // Empty handler for click outside of expand button
        }
    }

    fn create_preview(
        line: &str,
        format: FileFormat,
        parsed_getter: impl FnOnce() -> Option<Vec<String>>,
    ) -> String {
        let max_len = 150;
        match format {
            FileFormat::JSONL => {
                if line.chars().count() > max_len {
                    format!("{}...", line.chars().take(max_len).collect::<String>())
                } else {
                    line.to_string()
                }
            }
            FileFormat::CSV | FileFormat::TSV => {
                if let Some(fields) = parsed_getter() {
                    let joined = fields.join(" | ");
                    if joined.chars().count() > max_len {
                        format!("{}...", joined.chars().take(max_len).collect::<String>())
                    } else {
                        joined
                    }
                } else {
                    if line.chars().count() > max_len {
                        format!("{}...", line.chars().take(max_len).collect::<String>())
                    } else {
                        line.to_string()
                    }
                }
            }
        }
    }

    fn render_highlighted_text(&self, ui: &mut Ui, text: &str, search_term: &str) {
        let highlight_color = if self.dark_mode {
            egui::Color32::YELLOW
        } else {
            egui::Color32::from_rgb(255, 210, 0)
        };
        let highlight_text_color = if self.dark_mode {
            egui::Color32::BLACK
        } else {
            egui::Color32::BLACK
        };
        let normal_text_color = if self.dark_mode {
            egui::Color32::LIGHT_GRAY
        } else {
            egui::Color32::DARK_GRAY
        };
        let font_id =
            egui::FontId::monospace(ui.style().text_styles[&egui::TextStyle::Monospace].size);

        let mut layout = egui::text::LayoutJob::default();
        let term_len = search_term.len();

        if term_len == 0 {
            layout.append(
                text,
                0.0,
                egui::TextFormat::simple(font_id.clone(), normal_text_color),
            );
        } else {
            let mut last_end = 0;
            let text_lower;
            let term_lower;
            let search_haystack = if self.search.case_sensitive {
                text
            } else {
                text_lower = text.to_lowercase();
                &text_lower
            };
            let search_needle = if self.search.case_sensitive {
                search_term
            } else {
                term_lower = search_term.to_lowercase();
                &term_lower
            };

            for (start, _match) in search_haystack.match_indices(search_needle) {
                let end = start + term_len;
                if start > last_end {
                    layout.append(
                        &text[last_end..start],
                        0.0,
                        egui::TextFormat::simple(font_id.clone(), normal_text_color),
                    );
                }

                layout.append(
                    &text[start..end],
                    0.0,
                    egui::TextFormat {
                        font_id: font_id.clone(),
                        color: highlight_text_color,
                        background: highlight_color,
                        ..Default::default()
                    },
                );
                last_end = end;
            }

            if last_end < text.len() {
                layout.append(
                    &text[last_end..],
                    0.0,
                    egui::TextFormat::simple(font_id.clone(), normal_text_color),
                );
            }
        }

        ui.label(layout);
    }

    fn ensure_display_cached(&mut self, row: usize) {
        if self.display_cache.contains_key(&row) {
            return;
        }
        if self.format_reader.is_none() {
            return;
        }
        let reader = self.format_reader.as_ref().unwrap();

        let formatted_content = match reader.format() {
            FileFormat::JSONL => reader.get_record(row).and_then(|line| {
                serde_json::from_str::<Value>(&line)
                    .ok()
                    .and_then(|value| serde_json::to_string_pretty(&value).ok())
                    .or_else(|| Some(line))
            }),
            FileFormat::CSV | FileFormat::TSV => reader.get_record(row),
        };

        if let Some(content) = formatted_content {
            self.display_cache.insert(row, content);
        }

        let max_display_cache = if self.file_size > 10_000_000_000 {
            100
        } else {
            500
        };
        if self.display_cache.len() > max_display_cache {
            let keys_to_remove: Vec<_> = self
                .display_cache
                .keys()
                .take(self.display_cache.len() / 2)
                .copied()
                .collect();
            for key in keys_to_remove {
                self.display_cache.remove(&key);
            }
        }
    }
}

impl eframe::App for JsonlViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let render_start = Instant::now();

        for dropped_file in ctx.input(|i| i.raw.dropped_files.clone()) {
            if let Some(path) = dropped_file.path {
                if let Some(path_str) = path.to_str() {
                    let _ = self.open_file(path_str);
                }
            }
        }

        if let Some(receiver) = &self.format_reader_receiver {
            if let Ok(result) = receiver.try_recv() {
                match result {
                    Ok(reader) => {
                        self.format_reader = Some(reader);
                        if let Some(r) = &self.format_reader {
                            self.total_rows = r.total_records();
                            self.file_format = r.format();
                        }
                        self.show_message = Some((
                            format!(
                                "Loaded {} ~{} records ({})",
                                self.file_path
                                    .as_ref()
                                    .map(|p| Path::new(p)
                                        .file_name()
                                        .unwrap_or_default()
                                        .to_string_lossy())
                                    .unwrap_or_default(),
                                self.total_rows,
                                self.file_format.display_name()
                            ),
                            3.0,
                        ));
                        self.message_timer = None;
                    }
                    Err(e) => {
                        self.show_message = Some((format!("Error loading file: {}", e), 5.0));
                        self.message_timer = None;
                        self.file_path = None;
                    }
                }
                self.format_reader_receiver = None;
                self.previous_height_cache.clear();
            }
        }

        self.process_search_results();

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("📂 Open").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Text Files", &["jsonl", "csv", "tsv", "txt", "log"])
                        .add_filter("JSON Lines", &["jsonl"])
                        .add_filter("CSV", &["csv"])
                        .add_filter("TSV", &["tsv"])
                        .pick_file()
                    {
                        if let Some(path_str) = path.to_str() {
                            let _ = self.open_file(path_str);
                        }
                    }
                }

                if let Some(path) = &self.file_path {
                    let filename = Path::new(path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy();
                    let file_info = format!(
                        "{} ({}, ~{} lines)",
                        filename,
                        self.file_format.display_name(),
                        self.total_rows
                    );
                    ui.label(file_info).on_hover_text(path);
                } else {
                    ui.label("No file loaded.");
                }

                ui.separator();

                if ui
                    .button(if self.show_search {
                        "🔼 Hide Search"
                    } else {
                        "🔍 Search"
                    })
                    .clicked()
                {
                    self.show_search = !self.show_search;
                }
                if self.show_search {
                    ui.label("Find:");
                    let text_edit = ui.add(
                        egui::TextEdit::singleline(&mut self.search.term)
                            .desired_width(150.0)
                            .hint_text("Enter search term..."),
                    );

                    if text_edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        self.start_search();
                    }
                    if ui.button("Find").clicked() {
                        self.start_search();
                    }
                    ui.checkbox(&mut self.search.case_sensitive, "Aa");

                    if self.search.in_progress {
                        ui.spinner();
                        ui.label("Searching...");
                    } else if self.search.count > 0 {
                        ui.label(format!(
                            "({}/{})",
                            self.search.current_match_idx.map_or(0, |i| i + 1),
                            self.search.count
                        ));
                        if ui.button("↑").on_hover_text("Previous match").clicked() {
                            self.prev_match();
                        }
                        if ui.button("↓").on_hover_text("Next match").clicked() {
                            self.next_match();
                        }
                    } else if !self.search.term.is_empty() {
                        ui.label("(0 matches)");
                    }
                }

                ui.separator();

                if ui
                    .button(if self.dark_mode { "☀️" } else { "🌙" })
                    .on_hover_text("Toggle light/dark mode")
                    .clicked()
                {
                    self.dark_mode = !self.dark_mode;
                    ctx.set_visuals(if self.dark_mode {
                        egui::Visuals::dark()
                    } else {
                        egui::Visuals::light()
                    });
                }

                if self.total_rows > 0 {
                    if ui
                        .button("➕ Expand Vis.")
                        .on_hover_text("Expand visible rows")
                        .clicked()
                    {
                        let (start, end) = self.visible_range;
                        let limit = if self.file_size > 10_000_000_000 {
                            30
                        } else {
                            100
                        };
                        let actual_end = std::cmp::min(start + limit, end);
                        for row in start..actual_end {
                            if !self.expanded_rows.contains(&row) {
                                self.expanded_rows.insert(row);
                                self.ensure_display_cached(row);
                            }
                        }
                        self.previous_height_cache.clear();
                        self.show_message = Some((
                            format!("Expanded rows {} to {}", start + 1, actual_end),
                            2.0,
                        ));
                        self.message_timer = None;
                    }
                    if ui
                        .button("➖ Collapse All")
                        .on_hover_text("Collapse all expanded rows")
                        .clicked()
                    {
                        if !self.expanded_rows.is_empty() {
                            self.expanded_rows.clear();
                            self.previous_height_cache.clear();
                        }
                    }
                }

                if self.file_format == FileFormat::CSV || self.file_format == FileFormat::TSV {
                    ui.separator();
                    if ui
                        .checkbox(&mut self.text_wrapping, "Wrap Table Text")
                        .changed()
                    {
                        self.previous_height_cache.clear();
                    }
                    if self.text_wrapping {
                        ui.add(
                            egui::Slider::new(&mut self.max_column_width, 100.0..=800.0)
                                .text("Width"),
                        );
                    }
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.viewport_height = ui.available_height();

            if self.format_reader.is_none() && self.format_reader_receiver.is_some() {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.spinner();
                        ui.heading("Loading file...");
                        if self.file_size > 1_000_000_000 {
                            ui.label(format!(
                                "({:.1} GB)",
                                self.file_size as f64 / 1_073_741_824.0
                            ));
                        }
                    });
                });
            } else if self.format_reader.is_some() && self.total_rows > 0 {
                let reader = self.format_reader.as_ref().unwrap();

                let current_total_rows = reader.total_records();
                if current_total_rows != self.total_rows {
                    self.total_rows = current_total_rows;
                }

                let approx_total_height = self.height_before_row(self.total_rows);

                let mut scroll_area = egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .id_salt("main_scroll_area");

                if let Some(target_row) = self.scroll_to_search.take() {
                    let target_row_top = self.height_before_row(target_row);

                    let desired_offset = (target_row_top - self.viewport_height * 0.25).max(0.0);

                    self.scroll_memory = Some(desired_offset);
                    self.stabilize_counter = 5;

                    if !self.expanded_rows.contains(&target_row) {
                        self.expanded_rows.insert(target_row);
                        self.ensure_display_cached(target_row);
                        self.previous_height_cache.clear();
                    }
                }

                if self.stabilize_counter > 0 {
                    if let Some(scroll_pos) = self.scroll_memory {
                        scroll_area = scroll_area.vertical_scroll_offset(scroll_pos);
                        self.stabilize_counter -= 1;
                        if self.stabilize_counter == 0 {
                            self.scroll_memory = None;
                        }
                    } else {
                        self.stabilize_counter = 0;
                    }
                }

                let _scroll_output = scroll_area.show(ui, |ui| {
                    let scroll_offset = ui.clip_rect().top() - ui.min_rect().top();
                    self.last_scroll_offset = scroll_offset;

                    let view_top = (scroll_offset - BUFFER_HEIGHT).max(0.0);
                    let view_bottom = scroll_offset + self.viewport_height + BUFFER_HEIGHT;

                    let start_row = self.row_at_height(view_top);
                    let end_row = self.row_at_height(view_bottom) + 1; // +1

                    let clamped_end_row = end_row.min(self.total_rows); // Keep this clamp

                    self.visible_range = (start_row, clamped_end_row);

                    self.update_height_cache(start_row, clamped_end_row);

                    let space_before = self.height_before_row(start_row);
                    ui.add_space(space_before);

                    let mut calculated_rendered_height = 0.0;
                    for row in start_row..clamped_end_row {
                        let height = self.row_height(row);
                        self.render_row(ui, row, ctx);
                        ui.add_space(2.0);
                        calculated_rendered_height += height + 2.0;
                    }

                    let space_after =
                        approx_total_height - (space_before + calculated_rendered_height);
                    if space_after > 0.0 {
                        ui.add_space(space_after);
                    }
                });

                if self.stabilize_counter > 0 {
                    ctx.request_repaint();
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading("JSLON");
                        ui.label("High-Performance JSONL/CSV/TSV Viewer");
                        ui.add_space(20.0);
                        ui.label("Click 'Open' or drag & drop a file.");
                    });
                });
            }
        });

        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.total_rows > 0 {
                    ui.label(format!("Lines: {}", self.total_rows));
                    ui.separator();
                    ui.label(format!(
                        "Visible: {} - {}",
                        self.visible_range.0 + 1,
                        self.visible_range.1
                    ));
                    ui.separator();
                    ui.label(format!("Expanded: {}", self.expanded_rows.len()));
                } else if self.file_path.is_some() {
                    ui.label("Lines: 0");
                } else {
                    ui.label("No file loaded");
                }

                if self.search.count > 0 {
                    ui.separator();
                    let current = self.search.current_match_idx.map(|i| i + 1).unwrap_or(0);
                    ui.label(format!("Match: {}/{}", current, self.search.count));
                }

                ui.separator();
                ui.label(format!(
                    "Render: {:.1} ms",
                    self.render_duration.as_secs_f32() * 1000.0
                ));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some((msg, timestamp)) = &self.copy_message {
                        let current_time = ctx.input(|i| i.time);
                        if current_time - timestamp < 1.5 {
                            ui.label(RichText::new(msg).color(egui::Color32::GREEN));
                            ctx.request_repaint();
                        } else {
                            self.copy_message = None;
                        }
                    }

                    if let Some((message, duration)) = &self.show_message {
                        let now = ctx.input(|i| i.time);
                        let start_time = self.message_timer.get_or_insert(now);

                        if now - *start_time < *duration as f64 {
                            ui.label(message);
                            ctx.request_repaint();
                        } else {
                            self.show_message = None;
                            self.message_timer = None;
                        }
                    }
                });
            });
        });

        self.render_duration = render_start.elapsed();

        if self.format_reader_receiver.is_some()
            || self.search_receiver.is_some()
            || self.show_message.is_some()
            || self.copy_message.is_some()
            || self.stabilize_counter > 0
        {
            ctx.request_repaint();
        }
    }
}

fn main() -> Result<(), eframe::Error> {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([600.0, 400.0])
            .with_title("JSLON - High-Performance File Viewer"),
        ..Default::default()
    };

    eframe::run_native(
        "JSLON Viewer",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(JsonlViewer::new()))
        }),
    )
}
