
params.input = "/input/projects/dhs/smartsensingandimaging/development/fops"
params.day = null
params.tonemap = false
params.out_ext = null
params.dev = false
params.sample_n = params.dev ? 8 : -1
params.splits = 10 // how many jobs should each side-camera be split into
params.nproc = 20

assert params.day != null : "Usage: nextflow run . --day YYYY-MM-DD"
tonemap = params.tonemap ? "--tonemap" : ""
out_ext = params.out_ext ? "--out-ext $params.out_ext" : ""
out_path = params.tonemap ? "tonemapped" : "12-bit"

metadata_f = file("$params.input/$params.day/metadata_last.csv")
rows = Channel.from(metadata_f).splitCsv(header: true)
if(params.sample_n > 0){
    rows = rows.randomSample(params.sample_n, 1)
}

rows = rows.map { tuple(it.number as int % params.splits, it.side, it.camera, file("$it.day/$it.run/$it.camera/$it.image")) }

rows = rows.groupTuple(by: [0, 1, 2])

process export {
    time "60m"
    memory "80G"
    cpus params.nproc
    input:
        tuple val(split), val(side), val(camera), path("input/*") from rows
    output:
        tuple val(side), val(camera), path("output/*") into images
    """
    #!/bin/bash
    mkdir -p output
    mva_process --nproc $params.nproc convert --input input --output output $tonemap $out_ext
    """
}

process share {
    container "../../singularity/rclone.sif"
    time "3h"
    memory "1G"
    input:
        path("metadata.csv") from metadata_f
        path(rclone_cfg) from file("rclone.conf")
        tuple val(side), val(camera), path("input/*") from images
    """
    #!/bin/bash
    rclone --config $rclone_cfg copy -L --no-traverse metadata.csv CloudStor:/$out_path/$params.day
    rclone --config $rclone_cfg copy -L --no-traverse input CloudStor:/$out_path/$params.day/$side/$camera
    """
}
