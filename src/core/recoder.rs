use log4rs::{
    append::{
        console::{ConsoleAppender, Target},
        file::FileAppender,
    },
    config::{Appender, Config, Root},
    encode::pattern::PatternEncoder,
    filter::threshold::ThresholdFilter,
};

pub fn build_log_config(learner: &crate::core::parser::Learner) -> anyhow::Result<log4rs::Config> {
    let level = log::LevelFilter::Info;
    let mut log_path = std::path::PathBuf::from(&learner.recoder_home).join(&learner.name);
    if !log_path.exists() {
        std::fs::create_dir_all(&log_path)?;
    }

    log_path.push(chrono::Local::now().format("%Y%m-%d--%H:%M:%S").to_string() + ".log");

    // Build a stderr logger.
    let stderr = ConsoleAppender::builder().target(Target::Stderr).build();

    // Logging to log file.
    let logfile = FileAppender::builder()
        // Pattern: https://docs.rs/log4rs/*/log4rs/encode/pattern/index.html
        .encoder(Box::new(PatternEncoder::new(
            "{d(%+)(utc)} [{f}:{L}] {h({l})} -> {m}{n}",
        )))
        .build(log_path)
        .unwrap();

    // Log Trace level output to file where trace is the default level
    // and the programmatically specified level to stderr.
    let config = Config::builder()
        .appender(Appender::builder().build("logfile", Box::new(logfile)))
        .appender(
            Appender::builder()
                .filter(Box::new(ThresholdFilter::new(level)))
                .build("stderr", Box::new(stderr)),
        )
        .build(
            Root::builder()
                .appender("logfile")
                .appender("stderr")
                .build(log::LevelFilter::Trace),
        )
        .unwrap();

    Ok(config)
    // Use this to change log levels at runtime.
    // This means you can change the default log level to trace
    // if you are trying to debug an issue and need more logs on then turn it off
    // once you are done.
    // let handle = log4rs::init_config(config)?;
    //
    // log::error!("Goes to stderr and file");
    // log::warn!("Goes to stderr and file");
    // log::info!("Goes to stderr and file");
    // log::debug!("Goes to file only");
    // log::trace!("Goes to file only");
}
