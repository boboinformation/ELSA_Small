#!/usr/bin/env tclsh
# Quant — TensorRT FP16/INT8 quantization validate + bench.
#
# Usage:
#   ./script/Quant/run_quant.tcl
#   ./script/Quant/run_quant.tcl quant_mode=int8
#   ./script/Quant/run_quant.tcl task=bench checkpoint=/path/to/model_best.pth.tar
#   ./script/Quant/run_quant.tcl family=elsa_vit quant_mode=fp16 checkpoint=/path/...
#   ./script/Quant/run_quant.tcl dry_run=1

set script_dir [file dirname [file normalize [info script]]]
set root       [file normalize [file join $script_dir ../..]]
set runner_sh  [file join $script_dir run_quant.sh]

# ---- 預設值 (可被 key=value 覆寫) ---------------------------
array set cfg {
    quant_mode          fp16
    task                both
    family              elsa_swin
    model               ""
    checkpoint          ""
    device              cuda:0
    img_size            256
    val_batch           64
    bench_batch         1
    workers             4
    calib_batches       8
    warmup              50
    bench_iter          200
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
puts "[clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]  \[Quant\]"
puts "backend=tensorrt  quant=$cfg(quant_mode)  task=$cfg(task)  family=$cfg(family)"

# ---- 組合給 run_quant.sh 的環境變數 -------------------------
set env_pairs [list \
    RUN_QUANT_MODE=$cfg(quant_mode) \
    RUN_TASK=$cfg(task) \
    RUN_FAMILY=$cfg(family) \
    RUN_MODEL=$cfg(model) \
    RUN_CHECKPOINT=$cfg(checkpoint) \
    RUN_DEVICE=$cfg(device) \
    RUN_IMG_SIZE=$cfg(img_size) \
    RUN_VAL_BATCH=$cfg(val_batch) \
    RUN_BENCH_BATCH=$cfg(bench_batch) \
    RUN_WORKERS=$cfg(workers) \
    RUN_CALIB_BATCHES=$cfg(calib_batches) \
    RUN_WARMUP=$cfg(warmup) \
    RUN_BENCH_ITER=$cfg(bench_iter) \
    RUN_SPLIT=$cfg(split) \
    RUN_DRY_RUN=$cfg(dry_run)]

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
