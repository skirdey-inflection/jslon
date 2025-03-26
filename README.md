# JSLON

A blazing fast, memory-efficient viewer for large JSONL (JSON Lines) files.

## Features

- **Virtualized rendering** - Only visible rows are processed, making it efficient for gigabyte-sized files
- **Memory-mapped file access** - Files are loaded instantly regardless of size
- **Smart search** - Fast keyword search with term highlighting and navigation
- **Collapsible rows** - Expand only the entries you care about
- **Dark/Light mode** - Choose your preferred theme

## Requirements

- Rust 1.60 or later
- Cargo

## Building from Source

### Quick Start

```bash
# Clone the repository
git https://github.com/skirdey-inflection/jslon
cd jslon

# Build in debug mode
cargo build

# Run the application
cargo run
```

### Build Release Version

```bash
cargo build --release
```

The executable will be available at `target/release/jslon` (or `jslon.exe` on Windows).

## Packaging as Application

We use `cargo-bundle` for creating application bundles on all platforms.

### Install cargo-bundle

```bash
cargo install cargo-bundle
```


### Create Bundles

#### macOS

```bash
cargo bundle --release
```

This creates an `.app` bundle in `target/release/bundle/macos/`.

#### Linux

For Debian/Ubuntu:

```bash
cargo bundle --format deb
```

For RPM-based distributions:

```bash
cargo bundle --format rpm
```

The packages will be available in `target/release/bundle/`.

#### Windows

```bash
cargo bundle --format msi
```

This creates an MSI installer in `target/release/bundle/windows/`.

## Manual Execution

If you prefer not to create a bundle, you can run the release build directly:

### macOS/Linux

```bash
./target/release/jslon
```

### Windows

```bash
.\target\release\jslon.exe
```

## Usage Guide

1. **Opening Files**
   - Click the "📂 Open File" button and select a JSONL file
   - The file will be loaded instantly, even for very large files

2. **Navigation**
   - Use the scroll wheel to navigate through the file
   - Click "►" to expand a row and "▼" to collapse it
   - "Expand All" expands visible rows (more will expand as you scroll)
   - "Collapse All" collapses all rows

3. **Search**
   - Click "🔍 Search" to open the search bar
   - Enter your search term and press Enter or click "Find"
   - Use "↑ Prev" and "↓ Next" buttons to navigate between matches
   - Toggle "Case sensitive" for case-sensitive search
   - Search highlights matched terms specifically, not entire rows

4. **View Options**
   - Toggle between dark and light themes with the "🌙 Dark" / "☀ Light" button
