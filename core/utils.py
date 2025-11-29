def bytes_to_bits(data: bytes) -> list[int]:
    bits = []
    for byte in data:
        bit_string = bin(byte)[2:].zfill(8)
        bits.extend(int(b) for b in bit_string)

    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("Number of bits must be a multiple of 8.")

    result = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i + 8]
        byte_str = ''.join(str(b) for b in byte_bits)
        byte_int = int(byte_str, 2)
        result.append(byte_int)

    return bytes(result)