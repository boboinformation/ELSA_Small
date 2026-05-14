#!/usr/bin/env tclsh
# ReTrain — PyTorch backend, fp32 precision (no Triton kernel).
#
# Usage:
#   ./script/ReTrain/run_retrain.tcl
#   ./script/ReTrain/run_retrain.tcl task=bench
#   ./script/ReTrain/run_retrain.tcl task=validate checkpoint=/path/to/model.pth.tar
#   ./script/ReTrain/run_retrain.tcl task=train family=elsa_swin lr=6.25e-5
#   ./script/ReTrain/run_retrain.tcl task=train family=elsa_vit train_batch=300 channels_last=1
#   ./script/ReTrain/run_retrain.tcl family=sic_int
#   ./script/ReTrain/run_retrain.tcl dry_run=1

set script_dir [file dirname [file normalize [info script]]]
set root       [file normalize [file join $script_dir ../..]]
set runner_sh  [file join $script_dir run_retrain.sh]

# ---- 預設值 (可被 key=value 覆寫) ---------------------------
array set cfg {
    task                bench
    family              elsa_swin
    model               ""
    checkpoint          ""
    initial_checkpoint  ""
    device              cuda:0
    img_size            ""
    batch               1
    val_batch           64
    train_batch         64
    warmup              5
    trials              20
    workers             4
    channels_last       0
    lr                  ""
    lr_base             ""
    min_lr              ""
    weight_decay        ""
    warmup_epochs       ""
    epochs              300
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
puts "[clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]  \[ReTrain\]"
puts "backend=pytorch  dtype=fp32  task=$cfg(task)  family=$cfg(family)"

# ---- 組合給 run_retrain.sh 的環境變數 -----------------------
set env_pairs [list \
    RUN_TASK=$cfg(task) \
    RUN_FAMILY=$cfg(family) \
    RUN_MODEL=$cfg(model) \
    RUN_CHECKPOINT=$cfg(checkpoint) \
    RUN_INITIAL_CHECKPOINT=$cfg(initial_checkpoint) \
    RUN_DEVICE=$cfg(device) \
    RUN_BATCH=$cfg(batch) \
    RUN_VAL_BATCH=$cfg(val_batch) \
    RUN_TRAIN_BATCH=$cfg(train_batch) \
    RUN_WARMUP=$cfg(warmup) \
    RUN_TRIALS=$cfg(trials) \
    RUN_WORKERS=$cfg(workers) \
    RUN_CHANNELS_LAST=$cfg(channels_last) \
    RUN_LR=$cfg(lr) \
    RUN_LR_BASE=$cfg(lr_base) \
    RUN_MIN_LR=$cfg(min_lr) \
    RUN_WEIGHT_DECAY=$cfg(weight_decay) \
    RUN_WARMUP_EPOCHS=$cfg(warmup_epochs) \
    RUN_EPOCHS=$cfg(epochs) \
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
