"""L2 pixel policies (bit positions), distinct from L3 measurement warnings.

SPHEREx Explanatory Supplement v2, Table 16 and sections 3.3/3.4.3.
QR3 source/background defaults are deliberately conservative; the mosaic policy
follows the archive mosaic exclusions. Explicit caller overrides remain supported.
"""
from numbers import Integral

LEGACY_FATAL_BITS = (0, 1, 2, 4, 6, 7, 9, 10, 11, 14, 15, 17, 19, 22, 24, 26, 27, 28, 29)
QR3_SOURCE_BITS = (0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29)
QR3_BACKGROUND_BITS = tuple(sorted((*QR3_SOURCE_BITS, 21)))
QR3_MOSAIC_BITS = (0, 2, 6, 7, 9, 10, 11, 14, 15, 17, 20, 22, 23, 24, 25, 26, 27, 28, 29)
QR3_WARNING_BITS = QR3_SOURCE_BITS


def validate_bits(bits):
    if bits is not None and any(isinstance(b, bool) or not isinstance(b, Integral) or not 0 <= b < 32 for b in bits):
        raise ValueError('L2 FLAGS bit positions must be integers from 0 through 31; L3 bit 32 is not a pixel bit')


def bitmask(bits):
    bits = tuple(bits)
    validate_bits(bits)
    value = 0
    for bit in bits:
        value |= 1 << int(bit)
    return value
