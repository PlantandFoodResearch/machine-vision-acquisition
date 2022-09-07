# File sync server setup
A basic lsyncd setup to move files to a server

## Resuming an interrupted upload
If an upload is interrupted (e.g. machine reboot), it won't be continued. Since only *new* files are uploaded, you need to manually sync *all* files. This can be done by removing the `init      = false,` line from `/etc/lsyncd/lsyncd.conf.lua` (the installed config) and restarting the service. E.g. `sudoedit /etc/lsyncd/lsyncd.conf.lua` followed by `sudo systemctl restart lsyncd.service`

**Warning:** This will potentially take a long time and upload files removed from Powerplant but not from the sync folder. Consider first clearing out the sync folder.

## Notes on getting lsyncd working
https://www.howtoforge.com/how-to-synchronize-directories-using-lsyncd-on-ubuntu/

https://lsyncd.github.io/lsyncd/

On older ubuntu install from source https://github.com/lsyncd/lsyncd/archive/refs/tags/release-2.2.3.tar.gz (needs lua compiler and dev). otherwise: 
`apt-get install lsyncd`

```bash
mkdir /etc/lsyncd
touch /etc/lsyncd/lsyncd.conf.lua
# Contents should be like the example, but modified to suit
systemctl enable lsyncd && systemctl restart lsyncd
# Watch progress / triggers:
journalctl -f -t lsyncd
```