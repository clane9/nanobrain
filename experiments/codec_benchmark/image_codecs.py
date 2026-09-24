import gzip
import io
import zstandard as zstd

import nibabel as nib
import numpy as np

from nanobrain.utils.rle import brle_to_dense, dense_to_brle

RLE_DTYPE = np.uint8
ZSTD_LEVEL = 9


def pack_values(values: np.ndarray):
    values = values.astype(np.uint16)
    lo = (values & 0xFF).astype(np.uint8)
    hi = (values >> 8).astype(np.uint8)
    if len(hi) % 2:
        hi = np.append(hi, np.uint8(0))
    return lo, (hi[0::2] | (hi[1::2] << 4))


def unpack_values(lo: np.ndarray, packed_hi: np.ndarray):
    hi = np.empty(2 * len(packed_hi), dtype=np.uint8)
    hi[0::2] = packed_hi & 0x0F
    hi[1::2] = packed_hi >> 4
    hi = hi[: len(lo)]
    return (lo.astype(np.uint16) | (hi.astype(np.uint16) << 8)).astype(np.int16)


def encode_nifti(data, affine):
    return nib.Nifti1Image(data, affine).to_bytes()


def decode_nifti(buf):
    img = nib.Nifti1Image.from_bytes(buf)
    return np.asarray(img.dataobj, dtype=np.int16), img.affine


def encode_nii_gz(data, affine):
    return gzip.compress(encode_nifti(data, affine), 6)


def decode_nii_gz(buf):
    return decode_nifti(gzip.decompress(buf))


def encode_nii_zst(data, affine):
    return zstd.compress(encode_nifti(data, affine), 3)


def decode_nii_zst(buf):
    return decode_nifti(zstd.decompress(buf))


def encode_npz(data, affine):
    buf = io.BytesIO()
    np.savez(buf, data=data, affine=affine)
    return buf.getvalue()


def decode_npz(buf):
    npz = np.load(io.BytesIO(buf))
    return npz["data"], npz["affine"]


def encode_npz_zlib(data, affine):
    buf = io.BytesIO()
    np.savez_compressed(buf, data=data, affine=affine)
    return buf.getvalue()


def encode_npz_sparse(data, affine):
    mask = data != 0
    buf = io.BytesIO()
    np.savez(
        buf,
        mask=np.packbits(mask.ravel()),
        values=data[mask],
        shape=np.array(data.shape),
        affine=affine,
    )
    return buf.getvalue()


def decode_npz_sparse(buf):
    npz = np.load(io.BytesIO(buf))
    shape = tuple(int(size) for size in npz["shape"])
    mask = np.unpackbits(npz["mask"], count=int(np.prod(shape))).astype(bool)
    data = np.zeros(mask.size, dtype=np.int16)
    data[mask] = npz["values"]
    return data.reshape(shape), npz["affine"]


def encode_npz_sparse_rle(data, affine):
    mask = data != 0
    buf = io.BytesIO()
    np.savez(
        buf,
        mask=dense_to_brle(mask.ravel(), dtype=RLE_DTYPE),
        values=data[mask],
        shape=np.array(data.shape),
        affine=affine,
    )
    return buf.getvalue()


def decode_npz_sparse_rle(buf):
    npz = np.load(io.BytesIO(buf))
    shape = tuple(int(size) for size in npz["shape"])
    mask = brle_to_dense(npz["mask"])
    data = np.zeros(mask.size, dtype=np.int16)
    data[mask] = npz["values"]
    return data.reshape(shape), npz["affine"]


def encode_npz_sparse_rle_pack12(data, affine):
    mask = data != 0
    lo, packed_hi = pack_values(data[mask])
    buf = io.BytesIO()
    np.savez(
        buf,
        mask=dense_to_brle(mask.ravel(), dtype=RLE_DTYPE),
        lo=lo,
        packed_hi=packed_hi,
        shape=np.array(data.shape),
        affine=affine,
    )
    return buf.getvalue()


def decode_npz_sparse_rle_pack12(buf):
    npz = np.load(io.BytesIO(buf))
    shape = tuple(int(size) for size in npz["shape"])
    mask = brle_to_dense(npz["mask"])
    data = np.zeros(mask.size, dtype=np.int16)
    data[mask] = unpack_values(npz["lo"], npz["packed_hi"])
    return data.reshape(shape), npz["affine"]


def encode_npz_sparse_rle_zstd(data, affine):
    return zstd.compress(encode_npz_sparse_rle(data, affine), ZSTD_LEVEL)


def decode_npz_sparse_rle_zstd(buf):
    return decode_npz_sparse_rle(zstd.decompress(buf))


def encode_npz_sparse_rle_pack12_zstd(data, affine):
    return zstd.compress(encode_npz_sparse_rle_pack12(data, affine), ZSTD_LEVEL)


def decode_npz_sparse_rle_pack12_zstd(buf):
    return decode_npz_sparse_rle_pack12(zstd.decompress(buf))


CODECS = {
    "nii.gz": (encode_nii_gz, decode_nii_gz),
    "nii.zst": (encode_nii_zst, decode_nii_zst),
    "npz": (encode_npz, decode_npz),
    "npz-zlib": (encode_npz_zlib, decode_npz),
    "npz-sparse": (encode_npz_sparse, decode_npz_sparse),
    "npz-sparse-rle": (encode_npz_sparse_rle, decode_npz_sparse_rle),
    "npz-sparse-rle-pack12": (encode_npz_sparse_rle_pack12, decode_npz_sparse_rle_pack12),
    "npz-sparse-rle-zstd": (encode_npz_sparse_rle_zstd, decode_npz_sparse_rle_zstd),
    "npz-sparse-rle-pack12-zstd": (
        encode_npz_sparse_rle_pack12_zstd,
        decode_npz_sparse_rle_pack12_zstd,
    ),
}

try:
    import imagecodecs

    def encode_npz_jxl(data, affine):
        planes = np.ascontiguousarray(data.astype(np.uint16).transpose(2, 0, 1))
        blob = imagecodecs.jpegxl_encode(
            planes, lossless=True, effort=1, bitspersample=12, planar=True, numthreads=1
        )
        buf = io.BytesIO()
        np.savez(buf, jxl=np.frombuffer(blob, dtype=np.uint8), affine=affine)
        return buf.getvalue()

    def decode_npz_jxl(buf):
        npz = np.load(io.BytesIO(buf))
        planes = np.squeeze(imagecodecs.jpegxl_decode(npz["jxl"].tobytes(), numthreads=1))
        return planes.transpose(1, 2, 0).astype(np.int16), npz["affine"]

    CODECS["npz-jxl"] = (encode_npz_jxl, decode_npz_jxl)
except ImportError:
    pass
