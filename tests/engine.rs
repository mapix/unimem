use unimem::config::Config;
use unimem::decide::{ChainDecider, HeuristicDecider};
use unimem::embed::HashEmbedder;
use unimem::engine::{Engine, RecallReq, WriteOutcome, WriteReq};
use unimem::store::Store;

fn engine() -> Engine {
    Engine::with_parts(
        Config::default(),
        Store::open_memory().unwrap(),
        Box::new(HashEmbedder { dim: 512 }),
        ChainDecider::new(vec![Box::new(HeuristicDecider)]),
    )
}

fn git_project(dir: &std::path::Path, name: &str) -> String {
    let root = dir.join(name);
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::write(root.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
    std::fs::write(root.join(".git/config"), format!("[remote \"origin\"]\n\turl = git@github.com:me/{name}.git\n")).unwrap();
    std::fs::write(root.join("README.md"), format!("# {name}\n\n{name} is a small tool for testing uni-mem project detection.\n")).unwrap();
    root.display().to_string()
}

fn w(c: &str) -> WriteReq {
    WriteReq { content: c.into(), ..Default::default() }
}

#[tokio::test]
async fn write_scope_dedupe_supersede_recall() {
    let e = engine();
    let d = tempfile::tempdir().unwrap();
    let a = git_project(d.path(), "alpha");
    let b = git_project(d.path(), "beta");
    let ca = e.ctx_for("claude-code", "s1", Some(&a));
    let cb = e.ctx_for("codex", "s2", Some(&b));
    assert_eq!(ca.pid(), Some("github.com/me/alpha"));

    // preference without project entities → global
    let r = e.write(&ca, w("I prefer replies in Chinese and concise commit messages")).await.unwrap();
    let WriteOutcome::Stored { scope, kind, .. } = r else { panic!("{r:?}") };
    assert_eq!(scope, "global");
    assert_eq!(kind, "preference");

    // project fact → project scope
    let r = e.write(&ca, w("We deploy alpha through `ops/deploy.sh`, never by hand")).await.unwrap();
    let WriteOutcome::Stored { scope, .. } = r else { panic!("{r:?}") };
    assert_eq!(scope, "project:github.com/me/alpha");

    // exact duplicate
    let r = e.write(&ca, w("We deploy alpha through `ops/deploy.sh`, never by hand")).await.unwrap();
    assert!(matches!(r, WriteOutcome::Duplicate { .. }), "{r:?}");

    // status rot rejected
    let r = e.write(&ca, w("Next step is to refactor the parser module")).await.unwrap();
    assert!(matches!(r, WriteOutcome::Rejected { .. }), "{r:?}");

    // secret rejected
    let r = e.write(&ca, w("the api key is sk-ant-REDACTEDREDACTEDREDACTED123")).await.unwrap();
    assert!(matches!(r, WriteOutcome::Rejected { .. }), "{r:?}");

    // supersede: changed number
    let WriteOutcome::Stored { id: old, .. } = e.write(&ca, w("The alpha month-end job runs at 21:00 on the warehouse")).await.unwrap() else { panic!() };
    let r = e.write(&ca, WriteReq { content: "The alpha month-end job runs at 22:00 on the warehouse".into(), why: Some("the warehouse was downsized to Medium".into()), ..Default::default() }).await.unwrap();
    let WriteOutcome::Stored { superseded, .. } = r else { panic!("{r:?}") };
    assert_eq!(superseded, vec![old.clone()]);

    // recall from another harness in another project sees global, not alpha's project notes
    let out = e.recall(&cb, RecallReq { query: "how should replies be written, language?".into(), k: 5, ..Default::default() }).await.unwrap();
    assert!(out.notes.iter().any(|n| n.content.contains("Chinese")), "{out:?}");
    assert!(out.notes.iter().all(|n| !n.content.contains("deploy.sh")));

    // recall in alpha finds deploy + latest job time, not the superseded one
    let out = e.recall(&ca, RecallReq { query: "how do we deploy?".into(), k: 5, ..Default::default() }).await.unwrap();
    assert!(out.notes.first().is_some_and(|n| n.content.contains("deploy.sh")), "{out:?}");
    let out = e.recall(&ca, RecallReq { query: "when does the month-end job run".into(), k: 5, ..Default::default() }).await.unwrap();
    assert!(out.notes.first().is_some_and(|n| n.content.contains("22:00")), "{out:?}");
    // the outdated note may surface through history edges, but only flagged as superseded
    assert!(out.notes.iter().all(|n| n.id != old || n.superseded));

    // cross-project mention pulls the other project in
    let out = e.recall(&cb, RecallReq { query: "how does alpha deploy?".into(), k: 5, ..Default::default() }).await.unwrap();
    assert!(out.notes.iter().any(|n| n.content.contains("deploy.sh")), "{out:?}");

    // passive gate skips acks
    let out = e.recall(&ca, RecallReq { query: "好的".into(), passive: true, ..Default::default() }).await.unwrap();
    assert!(out.notes.is_empty());

    // explicit scope by project name from elsewhere
    let r = e.write(&cb, WriteReq { content: "alpha's CI uses GitHub Actions with a macOS runner".into(), scope: Some("alpha".into()), ..Default::default() }).await.unwrap();
    let WriteOutcome::Stored { scope, .. } = r else { panic!("{r:?}") };
    assert_eq!(scope, "project:github.com/me/alpha");

    // preamble renders project and preference
    let p = e.preamble(&ca).unwrap();
    assert!(p.contains("alpha") && p.contains("Chinese"), "{p}");
}

#[tokio::test]
async fn chinese_recall() {
    let e = engine();
    let ctx = e.ctx_for("claude-code", "s", None);
    e.write(&ctx, w("用户喜欢用 pnpm 而不是 npm 管理前端依赖")).await.unwrap();
    e.write(&ctx, w("发布流程：先跑 make test，然后执行 deploy/release.sh")).await.unwrap();
    let out = e.recall(&ctx, RecallReq { query: "前端依赖用什么包管理器".into(), k: 3, ..Default::default() }).await.unwrap();
    assert!(out.notes.first().is_some_and(|n| n.content.contains("pnpm")), "{out:?}");
}
