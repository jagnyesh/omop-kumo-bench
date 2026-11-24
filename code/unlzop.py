"""Minimal lzop (.lzo) container decoder using lzallright's LZO1X.
Usage: python unlzop.py in.csv.lzo out.csv"""
import struct, sys, zlib
from lzallright import LZOCompressor

MAGIC = b"\x89LZO\x00\r\n\x1a\n"

def decode(src, dst):
    c = LZOCompressor()
    with open(src, "rb") as f, open(dst, "wb") as out:
        assert f.read(9) == MAGIC, "not an lzop file"
        version, = struct.unpack(">H", f.read(2))
        f.read(2)  # lib version
        if version >= 0x0940:
            f.read(2)  # version needed
        method = f.read(1)[0]
        if version >= 0x0940:
            f.read(1)  # level
        flags, = struct.unpack(">I", f.read(4))
        if flags & 0x800:
            f.read(4)  # filter
        f.read(4)  # mode
        f.read(4)  # mtime low
        if version >= 0x0940:
            f.read(4)  # mtime high
        nlen = f.read(1)[0]
        f.read(nlen)
        f.read(4)  # header checksum
        if flags & 0x40:
            elen, = struct.unpack(">I", f.read(4)); f.read(elen + 4)
        total = 0
        while True:
            dlen, = struct.unpack(">I", f.read(4))
            if dlen == 0:
                break
            slen, = struct.unpack(">I", f.read(4))
            if flags & 0x1: f.read(4)
            if flags & 0x100: f.read(4)
            if slen < dlen:
                if flags & 0x2: f.read(4)
                if flags & 0x200: f.read(4)
            data = f.read(slen)
            block = data if slen == dlen else c.decompress(data, output_size_hint=dlen)
            assert len(block) == dlen, (len(block), dlen)
            out.write(block); total += dlen
    return total

if __name__ == "__main__":
    print(decode(sys.argv[1], sys.argv[2]))
