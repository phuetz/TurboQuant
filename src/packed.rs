use crate::error::{Result, TurboQuantError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedBits {
    words: Vec<u64>,
    bits_per_symbol: u8,
    symbol_count: usize,
}

impl PackedBits {
    pub fn zeros(symbol_count: usize, bits_per_symbol: u8) -> Result<Self> {
        validate_bits_per_symbol(bits_per_symbol)?;
        let total_bits = bits_per_symbol as usize * symbol_count;
        let word_count = total_bits.div_ceil(64);
        Ok(Self {
            words: vec![0; word_count],
            bits_per_symbol,
            symbol_count,
        })
    }

    pub fn pack_values(values: &[u64], bits_per_symbol: u8) -> Result<Self> {
        let mut packed = Self::zeros(values.len(), bits_per_symbol)?;
        for (index, &value) in values.iter().enumerate() {
            packed.set(index, value)?;
        }
        Ok(packed)
    }

    pub fn pack_signs(signs: &[i8]) -> Result<Self> {
        let values = signs
            .iter()
            .map(|&sign| if sign >= 0 { 1_u64 } else { 0_u64 })
            .collect::<Vec<_>>();
        Self::pack_values(&values, 1)
    }

    pub fn bits_per_symbol(&self) -> u8 {
        self.bits_per_symbol
    }

    pub fn symbol_count(&self) -> usize {
        self.symbol_count
    }

    pub fn storage_bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<u64>()
    }

    pub fn raw_words(&self) -> &[u64] {
        &self.words
    }

    pub fn get(&self, index: usize) -> Result<u64> {
        if index >= self.symbol_count {
            return Err(TurboQuantError::PackedIndexOutOfRange {
                index,
                symbol_count: self.symbol_count,
            });
        }
        if self.bits_per_symbol == 0 {
            return Ok(0);
        }

        let start_bit = index * self.bits_per_symbol as usize;
        let word_index = start_bit / 64;
        let bit_offset = start_bit % 64;
        let mask = symbol_mask(self.bits_per_symbol);

        if bit_offset + self.bits_per_symbol as usize <= 64 {
            Ok((self.words[word_index] >> bit_offset) & mask)
        } else {
            let low_bits = 64 - bit_offset;
            let high_bits = self.bits_per_symbol as usize - low_bits;
            let low_mask = (1_u64 << low_bits) - 1;
            let low = (self.words[word_index] >> bit_offset) & low_mask;
            let high = self.words[word_index + 1] & ((1_u64 << high_bits) - 1);
            Ok(low | (high << low_bits))
        }
    }

    pub fn get_sign(&self, index: usize) -> Result<i8> {
        Ok(if self.get(index)? == 0 { -1 } else { 1 })
    }

    pub fn unpack_values(&self) -> Result<Vec<u64>> {
        (0..self.symbol_count)
            .map(|index| self.get(index))
            .collect()
    }

    /// Bulk-unpack all symbols into an existing slice, avoiding per-element
    /// bounds checks and Result wrapping on the hot path.
    pub fn unpack_values_into(&self, output: &mut [u64]) -> Result<()> {
        if output.len() < self.symbol_count {
            return Err(TurboQuantError::PackedIndexOutOfRange {
                index: self.symbol_count,
                symbol_count: output.len(),
            });
        }
        if self.bits_per_symbol == 0 {
            output[..self.symbol_count].fill(0);
            return Ok(());
        }
        let mask = symbol_mask(self.bits_per_symbol);
        let bps = self.bits_per_symbol as usize;
        for index in 0..self.symbol_count {
            let start_bit = index * bps;
            let word_index = start_bit / 64;
            let bit_offset = start_bit % 64;
            output[index] = if bit_offset + bps <= 64 {
                (self.words[word_index] >> bit_offset) & mask
            } else {
                let low_bits = 64 - bit_offset;
                let low = (self.words[word_index] >> bit_offset) & ((1_u64 << low_bits) - 1);
                let high =
                    self.words[word_index + 1] & ((1_u64 << (bps - low_bits)) - 1);
                low | (high << low_bits)
            };
        }
        Ok(())
    }

    /// Bulk-unpack all symbols as signs (-1 or +1) into an existing slice.
    pub fn unpack_signs_into(&self, output: &mut [i8]) -> Result<()> {
        if output.len() < self.symbol_count {
            return Err(TurboQuantError::PackedIndexOutOfRange {
                index: self.symbol_count,
                symbol_count: output.len(),
            });
        }
        if self.bits_per_symbol == 0 {
            output[..self.symbol_count].fill(-1);
            return Ok(());
        }
        let bps = self.bits_per_symbol as usize;
        for index in 0..self.symbol_count {
            let start_bit = index * bps;
            let word_index = start_bit / 64;
            let bit_offset = start_bit % 64;
            let val = (self.words[word_index] >> bit_offset) & 1;
            output[index] = if val == 0 { -1 } else { 1 };
        }
        Ok(())
    }

    pub fn unpack_signs(&self) -> Result<Vec<i8>> {
        (0..self.symbol_count)
            .map(|index| self.get_sign(index))
            .collect()
    }

    fn set(&mut self, index: usize, value: u64) -> Result<()> {
        if index >= self.symbol_count {
            return Err(TurboQuantError::PackedIndexOutOfRange {
                index,
                symbol_count: self.symbol_count,
            });
        }

        let mask = symbol_mask(self.bits_per_symbol);
        if value > mask {
            return Err(TurboQuantError::PackedValueOutOfRange {
                value,
                bits_per_symbol: self.bits_per_symbol,
            });
        }
        if self.bits_per_symbol == 0 {
            return Ok(());
        }

        let start_bit = index * self.bits_per_symbol as usize;
        let word_index = start_bit / 64;
        let bit_offset = start_bit % 64;

        if bit_offset + self.bits_per_symbol as usize <= 64 {
            self.words[word_index] &= !(mask << bit_offset);
            self.words[word_index] |= value << bit_offset;
        } else {
            let low_bits = 64 - bit_offset;
            let high_bits = self.bits_per_symbol as usize - low_bits;
            let low_mask = (1_u64 << low_bits) - 1;
            let low = value & low_mask;
            let high = value >> low_bits;

            self.words[word_index] &= !(low_mask << bit_offset);
            self.words[word_index] |= low << bit_offset;

            let high_mask = (1_u64 << high_bits) - 1;
            self.words[word_index + 1] &= !high_mask;
            self.words[word_index + 1] |= high;
        }

        Ok(())
    }
}

fn validate_bits_per_symbol(bits_per_symbol: u8) -> Result<()> {
    if bits_per_symbol <= 63 {
        Ok(())
    } else {
        Err(TurboQuantError::UnsupportedBitWidth(bits_per_symbol))
    }
}

fn symbol_mask(bits_per_symbol: u8) -> u64 {
    if bits_per_symbol == 0 {
        0
    } else {
        (1_u64 << bits_per_symbol) - 1
    }
}
