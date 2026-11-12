//! 검색 결과를 콘솔에서 확인하는 점검 도구.  locall-search [--kind doc] [--image 사진경로] 검색어…
//!
//! 앱과 같은 경로(임베딩 작업자 → Searcher)를 그대로 탄다.

use anyhow::Result;
use clap::Parser;
use locall::config::{default_db_path, load_config};
use locall::embed::Embedder;
use locall::search::Searcher;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "locall-search", about = "LocalL 검색 점검 도구")]
struct Args {
    #[arg(long, default_value = "all", help = "종류: all doc image dir video audio app archive")]
    kind: String,

    #[arg(short, default_value_t = 8, help = "보여 줄 결과 수")]
    n: usize,

    #[arg(long, help = "이 사진과 비슷한 사진 찾기")]
    image: Option<PathBuf>,

    #[arg(long, help = "모델 없이 키워드 일치만으로 찾기 (모델 적재 전 동작 확인용)")]
    no_model: bool,

    queries: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = load_config(None)?;

    let t = Instant::now();
    let searcher = Searcher::new(&cfg, &default_db_path())?;
    let (files, vectors) = searcher.stats();
    eprintln!("[색인] 파일 {} · 벡터 {} ({:.1}초)", files, vectors, t.elapsed().as_secs_f64());

    let mut embedder = if args.no_model {
        None
    } else {
        let t = Instant::now();
        let e = Embedder::spawn(&cfg)?;
        eprintln!("[모델] {} ({:.1}초)", e.device, t.elapsed().as_secs_f64());
        Some(e)
    };

    if let (Some(path), Some(e)) = (&args.image, embedder.as_mut()) {
        println!("== [사진] {}", path.display());
        for r in searcher.search_image(&e.image(path)?, args.n)? {
            println!("  {:.3} [{}] {}", r.score, r.match_type, r.path);
        }
    }

    for q in &args.queries {
        let t = Instant::now();
        let vec = match embedder.as_mut() {
            Some(e) => Some(e.query(q)?),
            None => None,
        };
        let embed_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let results = searcher.search(q, vec.as_deref(), &args.kind, args.n)?;
        println!("== {}  (임베딩 {:.0}ms · 검색 {:.0}ms)", q, embed_ms, t.elapsed().as_secs_f64() * 1000.0);
        for r in results {
            println!("  {:.3} [{}] {}", r.score, r.match_type, r.path);
        }
    }
    Ok(())
}
