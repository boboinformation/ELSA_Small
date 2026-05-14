#!/usr/bin/env tclsh
# Drop_In — 驗證 ELSA 作為 drop-in replacement 的準確率與推理速度（不需訓練）。
#
# Usage:
#   ./script/Drop_In/run_drop_in.tcl
#   ./script/Drop_In/run_drop_in.tcl task=validate
#   ./script/Drop_In/run_drop_in.tcl task=bench
#   ./script/Drop_In/run_drop_in.tcl family=elsa_vit
#   ./script/Drop_In/run_drop_in.tcl family=sic_int
#   ./script/Drop_In/run_drop_in.tcl family=elsa_swin checkpoint=/raid/jess/SICNet/swinv2_small_patch4_window8_256.pth
#   ./script/Drop_In/run_drop_in.tcl dry_run=1

set script_dir [file dirname [file normalize [info script]]]
set root       [file normalize [file join $script_dir ../..]]
set runner_sh  [file join $script_dir run_drop_in.sh]

# ---- 預設值 (可被 key=value 覆寫) ---------------------------
array set cfg {
    task                both
    family              elsa_swin
    model               ""
    checkpoint          ""
    device              cuda:0
    img_size            ""
    val_batch           64
    bench_batch         1
    warmup              5
    trials              20
    workers             4
    split               val
    dry_run             0
}

# ---- 解析 key=value ------------------------------------------
foreach arg $argv {
    if {![regexp {^([^=]+)=(.*)$} $arg -> key value]} {
        puts stderr "Invalid argument '$arg'. Use key=value."
        exit 2
    }
    if {![info exists cfg($key)]} {
        puts stderr "Unknown key '$key'. Valid keys: [array names cfg]"
        exit 2
    }
    set cfg($key) $value
}

# ---- 印出摘要 -----------------------------------------------
puts "[clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]  \[Drop_In\]"
set ckpt_label [expr {$cfg(checkpoint) eq "" ? "pretrained(HF)" : $cfg(checkpoint)}]
puts "task=$cfg(task)  family=$cfg(family)  checkpoint=$ckpt_label"

# ---- 組合給 run_drop_in.sh 的環境變數 -----------------------
set env_pairs [list \
    RUN_TASK=$cfg(task) \
    RUN_FAMILY=$cfg(family) \
    RUN_MODEL=$cfg(model) \
    RUN_CHECKPOINT=$cfg(checkpoint) \
    RUN_DEVICE=$cfg(device) \
    RUN_VAL_BATCH=$cfg(val_batch) \
    RUN_BENCH_BATCH=$cfg(bench_batch) \
    RUN_WARMUP=$cfg(warmup) \
    RUN_TRIALS=$cfg(trials) \
    RUN_WORKERS=$cfg(workers) \
    RUN_SPLIT=$cfg(split) \
    RUN_DRY_RUN=$cfg(dry_run)]

# img_size 只在明確設定時才傳（否則讓 sh 依 family 自動決定）
if {$cfg(img_size) ne ""} {
    lappend env_pairs RUN_IMG_SIZE=$cfg(img_size)
}

set cmd [concat [list env] $env_pairs [list bash $runner_sh]]
puts "command: [join $cmd " "]"

if {$cfg(dry_run)} { exit 0 }

if {[catch {exec {*}$cmd >@ stdout 2>@ stderr} result opts]} {
    set ec [dict get $opts -errorcode]
    if {[lindex $ec 0] eq "CHILDSTATUS"} {
        exit [lindex $ec 2]
    }
    puts stderr $result
    exit 1
}
