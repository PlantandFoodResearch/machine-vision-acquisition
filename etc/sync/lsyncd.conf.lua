settings {
}
sync {
    default.rsync,
    source    = "/home/user/workspace/cfnmxl/machine-vision-acquisition/tmp/data-root/",
    target    = "powerplant.pfr.co.nz:/input/projects/dhs/smartsensingandimaging/development",
    -- # Large delay to allow complete writing and also deletion to not interfere
    delay     = 60,
    -- # Do not remove remote files
    delete    = false,
    -- # only monitor going forwards, not existing
    init      = false,
    rsync     = {
        archive  = true,
        compress = true,
        chown = "cfnmxl:powerplant",
        prune_empty_dirs = true,
        -- This needs to be a valid key, not encrypted for now
        rsh = "/usr/bin/ssh -i /home/user/.ssh/mflange-edge-fops.nopass.key -oStrictHostKeyChecking=no -l cfnmxl",
        -- # Remove source files
        _extra = { "--remove-source-files" }
    }
}