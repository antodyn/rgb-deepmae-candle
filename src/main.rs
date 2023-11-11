mod core;
mod datasets;
mod models;
use clap::Parser;

fn main() -> anyhow::Result<()> {
    let lnr = core::Learner::parse();
    let _log_handle = log4rs::init_config(core::build_log_config(&lnr)?)?;

    core::run(&lnr)?;
    // core::rund(&lnr)?;

    println!("{:?}", lnr);
    Ok(())
}
