# Notes on getting lsyncd working
https://www.howtoforge.com/how-to-synchronize-directories-using-lsyncd-on-ubuntu/

https://lsyncd.github.io/lsyncd/

On older ubuntu install from source https://github.com/lsyncd/lsyncd/archive/refs/tags/release-2.2.3.tar.gz (needs lua compiler and dev). otherwise: 
`apt-get install lsyncd`

```bash
mkdir /etc/lsyncd
touch /etc/lsyncd/lsyncd.conf.lua
# Contents should be like the example, but modified to suit
systemctl enable lsyncd && systemctl restart lsyncd
```