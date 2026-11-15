//! Whole-timeline benchmark: plays an effect tick by tick the way the overlay
//! does (one 1/120 s step, then VT parse, then raster) and reports the tick
//! time distribution, not just an average, so stutter shows up.
//!
//! Configured with environment variables (all optional):
//!   SSAVER_BENCH_EFFECT          effect name (default decrypt)
//!   SSAVER_BENCH_WIDTH/HEIGHT    surface size in logical px (default 1920x1080)
//!   SSAVER_BENCH_SCALE           output scale (default 1)
//!   SSAVER_BENCH_SECONDS         how long to play, in seconds of 120 Hz ticks (default 10)
//!   SSAVER_BENCH_SEED            seed (default 1; "entropy" = no seed, as in production)
//!   SSAVER_BENCH_WINDOW_SECONDS  window for the appearance-growth figure (default 64)
//!
//! It plays for exactly the given duration: when the effect ends, the overlay's
//! own rebuild happens inside `advance_deferred`, and that tick is measured too.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::HashSet,
    hint::black_box,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::{Duration, Instant},
};

use omarchy_launch_screensaver::{
    engine::{Animation, FRAME_RATE},
    options::Options,
    raster::Rasterizer,
    vt::Cell,
};

const BUDGET: Duration = Duration::from_nanos(8_333_333);

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(true);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE: AtomicU64 = AtomicU64::new(0);

fn live_bytes() -> u64 {
    ALLOCATED_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(DEALLOCATED_BYTES.load(Ordering::Relaxed))
}

fn note_peak() {
    let live = live_bytes();
    PEAK_LIVE.fetch_max(live, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            note_peak();
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if COUNTING.load(Ordering::Relaxed) {
            DEALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() && COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
            DEALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            note_peak();
        }
        replacement
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `f` with allocation counting off (the bench's own bookkeeping).
/// Memory freed later by such allocations must also be freed uncounted.
fn uncounted<T>(f: impl FnOnce() -> T) -> T {
    COUNTING.store(false, Ordering::Relaxed);
    let out = f();
    COUNTING.store(true, Ordering::Relaxed);
    out
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name}: cannot parse '{value}'")),
        Err(_) => default,
    }
}

#[derive(Default, Clone, Copy)]
struct Tick {
    step: Duration,
    parse: Duration,
    raster: Duration,
    changed: usize,
}

impl Tick {
    fn total(&self) -> Duration {
        self.step + self.parse + self.raster
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let index = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[index]
}

fn summary(name: &str, mut values: Vec<Duration>) -> String {
    values.sort_unstable();
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    format!(
        "{name}_ms(p50={:.3} p95={:.3} p99={:.3} max={:.3})",
        ms(percentile(&values, 0.50)),
        ms(percentile(&values, 0.95)),
        ms(percentile(&values, 0.99)),
        ms(*values.last().unwrap_or(&Duration::ZERO)),
    )
}

type Appearance = (char, [u8; 3], [u8; 3]);

fn main() {
    let effect: String = env_or("SSAVER_BENCH_EFFECT", "decrypt".to_owned());
    let width: i32 = env_or("SSAVER_BENCH_WIDTH", 1920);
    let height: i32 = env_or("SSAVER_BENCH_HEIGHT", 1080);
    let scale: i32 = env_or("SSAVER_BENCH_SCALE", 1);
    let seconds: f64 = env_or("SSAVER_BENCH_SECONDS", 10.0);
    let window_seconds: f64 = env_or("SSAVER_BENCH_WINDOW_SECONDS", 64.0);
    let seed_text: String = env_or("SSAVER_BENCH_SEED", "1".to_owned());
    let seed = if seed_text == "entropy" {
        None
    } else {
        Some(
            seed_text
                .parse::<u64>()
                .expect("SSAVER_BENCH_SEED: a number or 'entropy'"),
        )
    };

    let mut rasterizer = Rasterizer::load().expect("load monospace font");
    let physical_width = width * scale;
    let physical_height = height * scale;
    let (cols, rows) = rasterizer.grid_size(physical_width, physical_height, scale);
    let options = Options {
        effect: Some(effect.clone()),
        seed,
        ..Options::default()
    };
    let baseline = live_bytes();
    let mut animation = match Animation::new(
        include_str!("../assets/screensaver.txt").to_owned(),
        options,
        cols,
        rows,
    ) {
        Ok(animation) => animation,
        Err(error) => {
            eprintln!("timeline-bench: cannot start effect '{effect}': {error:#}");
            std::process::exit(2);
        }
    };
    let mut pixels = vec![0u8; physical_width as usize * physical_height as usize * 4];
    let mut previous_cells: Vec<Cell> = Vec::new();

    let ticks_per_second = FRAME_RATE as f64;
    let ticks = (seconds * ticks_per_second).round() as usize;
    let window_ticks = ((window_seconds * ticks_per_second).round() as usize).max(1);
    let (mut seen, mut appearances_by_second, mut retained_by_second) = uncounted(|| {
        (
            HashSet::<Appearance>::with_capacity(1 << 16),
            Vec::<usize>::with_capacity(seconds as usize + 2),
            Vec::<u64>::with_capacity(seconds as usize + 2),
        )
    });
    let mut samples = uncounted(|| Vec::<Tick>::with_capacity(ticks));

    let allocations_before = ALLOCATIONS.load(Ordering::Relaxed);
    let wall = Instant::now();
    for tick in 0..ticks {
        let started = Instant::now();
        animation.advance_deferred().expect("advance ttfx");
        let stepped = Instant::now();
        animation.rasterize_pending().expect("parse frame");
        let parsed = Instant::now();
        let current = animation.cells();
        let first = previous_cells.len() != current.len();
        if first {
            rasterizer
                .render(
                    &mut pixels,
                    physical_width,
                    physical_height,
                    current,
                    cols,
                    rows,
                    scale,
                )
                .expect("raster frame");
        } else {
            rasterizer
                .render_delta(
                    &mut pixels,
                    physical_width,
                    physical_height,
                    current,
                    &previous_cells,
                    cols,
                    rows,
                    scale,
                )
                .expect("raster frame");
        }
        let rastered = Instant::now();

        let changed = uncounted(|| {
            let changed = if first {
                current.len()
            } else {
                current
                    .iter()
                    .zip(previous_cells.iter())
                    .filter(|(a, b)| a.ch != b.ch || a.fg != b.fg || a.bg != b.bg)
                    .count()
            };
            if first {
                previous_cells.resize(current.len(), Cell::default());
            }
            previous_cells.copy_from_slice(current);
            for cell in current {
                seen.insert((cell.ch, cell.fg, cell.bg));
            }
            if (tick + 1) % FRAME_RATE as usize == 0 {
                appearances_by_second.push(seen.len());
                retained_by_second.push(live_bytes().saturating_sub(baseline));
            }
            samples.push(Tick {
                step: stepped - started,
                parse: parsed - stepped,
                raster: rastered - parsed,
                changed,
            });
            changed
        });
        black_box(changed);
    }
    let elapsed = wall.elapsed();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocations_before;
    black_box(&pixels);

    COUNTING.store(false, Ordering::Relaxed);
    let totals: Vec<Duration> = samples.iter().map(Tick::total).collect();
    let over = totals.iter().filter(|d| **d > BUDGET).count();
    let mut longest_run = 0;
    let mut run = 0;
    for total in &totals {
        run = if *total > BUDGET { run + 1 } else { 0 };
        longest_run = longest_run.max(run);
    }
    let cells = (cols * rows) as f64;
    let mut changed: Vec<f64> = samples
        .iter()
        .skip(1)
        .map(|s| s.changed as f64 / cells)
        .collect();
    changed.sort_by(|a, b| a.total_cmp(b));
    let changed_min = changed.first().copied().unwrap_or(0.0);
    let changed_p50 = changed.get(changed.len() / 2).copied().unwrap_or(0.0);
    let appearances = seen.len();
    let window_start = ticks.saturating_sub(window_ticks) / FRAME_RATE as usize;
    let before_window = window_start
        .checked_sub(1)
        .and_then(|i| appearances_by_second.get(i).copied())
        .unwrap_or(0);
    let growth_pct = if appearances == 0 {
        0.0
    } else {
        100.0 * (appearances - before_window) as f64 / appearances as f64
    };
    let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
    let retained = live_bytes().saturating_sub(baseline);
    let peak = PEAK_LIVE.load(Ordering::Relaxed).saturating_sub(baseline);

    println!(
        "timeline-bench effect={effect} seed={seed_text} size={physical_width}x{physical_height} scale={scale} cells={cols}x{rows} ticks={ticks} wall_ms={:.1} {} {} {} {} over_8.3ms={over} longest_over_run={longest_run} allocations={allocations} changed_frac(p50={changed_p50:.3} min={changed_min:.3}) retained_mb={:.2} peak_mb={:.2} appearances={appearances} appearances_window_growth_pct={growth_pct:.2}",
        elapsed.as_secs_f64() * 1000.0,
        summary("step", samples.iter().map(|s| s.step).collect()),
        summary("parse", samples.iter().map(|s| s.parse).collect()),
        summary("raster", samples.iter().map(|s| s.raster).collect()),
        summary("total", totals.clone()),
        mb(retained),
        mb(peak),
    );
    let mut slowest: Vec<(usize, Duration)> = totals.iter().copied().enumerate().collect();
    slowest.sort_by_key(|&(_, total)| std::cmp::Reverse(total));
    let slowest: Vec<String> = slowest
        .iter()
        .take(10)
        .map(|(tick, total)| {
            format!(
                "{:.3}s:{:.3}ms",
                *tick as f64 / ticks_per_second,
                total.as_secs_f64() * 1000.0
            )
        })
        .collect();
    println!("slowest {}", slowest.join(" "));
    let series: Vec<String> = appearances_by_second
        .iter()
        .zip(retained_by_second.iter())
        .enumerate()
        .filter(|(second, _)| (second + 1) % 10 == 0)
        .map(|(second, (count, bytes))| format!("{}s:{}:{:.2}MB", second + 1, count, mb(*bytes)))
        .collect();
    println!("series {}", series.join(" "));
}
