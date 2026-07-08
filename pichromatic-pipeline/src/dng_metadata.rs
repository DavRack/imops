#[derive(Debug, Clone, Default)]
pub struct DngMetadata {
    pub dng_version: Option<[u8; 4]>,
    pub dng_backward_version: Option<[u8; 4]>,
    pub unique_camera_model: Option<String>,
    pub color_matrix1: Option<Vec<f32>>,
    pub color_matrix2: Option<Vec<f32>>,
    pub camera_calibration1: Option<Vec<f32>>,
    pub camera_calibration2: Option<Vec<f32>>,
    pub analog_balance: Option<Vec<f32>>,
    pub as_shot_neutral: Option<Vec<f32>>,
    pub baseline_exposure: Option<f32>,
    pub linear_response_limit: Option<f32>,
    pub shadow_scale: Option<f32>,
    pub noise_profile: Option<Vec<f64>>,
    pub profile_name: Option<String>,
    pub profile_tone_curve: Option<Vec<f32>>,
    pub lens_info: Option<Vec<f32>>,
    pub camera_serial_number: Option<String>,
    pub opcode_list1: Option<Vec<u8>>,
    pub opcode_list2: Option<Vec<u8>>,
    pub opcode_list3: Option<Vec<u8>>,
}

pub struct DngMetadataParser<'a> {
    data: &'a [u8],
    is_little_endian: bool,
}

impl<'a> DngMetadataParser<'a> {
    pub fn new(data: &'a [u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }
        let byte_order = &data[0..2];
        let is_little_endian = match byte_order {
            b"II" => true,
            b"MM" => false,
            _ => return None,
        };
        let magic = Self::read_u16_at(data, 2, is_little_endian);
        if magic != 42 {
            return None;
        }
        Some(Self { data, is_little_endian })
    }

    fn read_u16_at(data: &[u8], offset: usize, is_le: bool) -> u16 {
        if offset + 2 > data.len() {
            return 0;
        }
        let bytes = &data[offset..offset+2];
        if is_le {
            u16::from_le_bytes([bytes[0], bytes[1]])
        } else {
            u16::from_be_bytes([bytes[0], bytes[1]])
        }
    }

    fn read_u32_at(data: &[u8], offset: usize, is_le: bool) -> u32 {
        if offset + 4 > data.len() {
            return 0;
        }
        let bytes = &data[offset..offset+4];
        if is_le {
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        } else {
            u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        }
    }

    pub fn parse(&self) -> DngMetadata {
        let mut metadata = DngMetadata::default();
        let first_ifd_offset = Self::read_u32_at(self.data, 4, self.is_little_endian) as usize;
        let mut visited = std::collections::HashSet::new();
        self.parse_ifd(first_ifd_offset, &mut metadata, &mut visited);
        metadata
    }

    fn parse_ifd(&self, offset: usize, metadata: &mut DngMetadata, visited: &mut std::collections::HashSet<usize>) {
        if offset == 0 || offset >= self.data.len() || visited.contains(&offset) {
            return;
        }
        visited.insert(offset);

        let num_entries = Self::read_u16_at(self.data, offset, self.is_little_endian) as usize;
        let mut entry_offset = offset + 2;

        for _ in 0..num_entries {
            if entry_offset + 12 > self.data.len() {
                break;
            }

            let tag = Self::read_u16_at(self.data, entry_offset, self.is_little_endian);
            let typ = Self::read_u16_at(self.data, entry_offset + 2, self.is_little_endian);
            let count = Self::read_u32_at(self.data, entry_offset + 4, self.is_little_endian) as usize;
            let val_offset = Self::read_u32_at(self.data, entry_offset + 8, self.is_little_endian) as usize;

            let val_bytes: [u8; 4] = [
                self.data[entry_offset + 8],
                self.data[entry_offset + 9],
                self.data[entry_offset + 10],
                self.data[entry_offset + 11],
            ];

            match tag {
                330 => { // SubIFDs
                    let offsets = self.read_offsets(typ, count, val_offset, &val_bytes);
                    for sub_offset in offsets {
                        self.parse_ifd(sub_offset, metadata, visited);
                    }
                }
                34665 | 34853 => { // ExifIFD / GPSIFD — follow these linked IFDs
                    self.parse_ifd(val_offset, metadata, visited);
                }
                50706 => { // DNGVersion
                    if typ == 1 && count == 4 {
                        let bytes = self.read_bytes(count, val_offset, &val_bytes);
                        if bytes.len() == 4 {
                            metadata.dng_version = Some([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        }
                    }
                }
                50707 => { // DNGBackwardVersion
                    if typ == 1 && count == 4 {
                        let bytes = self.read_bytes(count, val_offset, &val_bytes);
                        if bytes.len() == 4 {
                            metadata.dng_backward_version = Some([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        }
                    }
                }
                50708 => { // UniqueCameraModel
                    if typ == 2 {
                        metadata.unique_camera_model = self.read_ascii(count, val_offset, &val_bytes);
                    }
                }
                50721 => { // ColorMatrix1
                    metadata.color_matrix1 = Some(self.read_float_array(typ, count, val_offset, &val_bytes));
                }
                50722 => { // ColorMatrix2
                    metadata.color_matrix2 = Some(self.read_float_array(typ, count, val_offset, &val_bytes));
                }
                50723 => { // CameraCalibration1
                    metadata.camera_calibration1 = Some(self.read_float_array(typ, count, val_offset, &val_bytes));
                }
                50724 => { // CameraCalibration2
                    metadata.camera_calibration2 = Some(self.read_float_array(typ, count, val_offset, &val_bytes));
                }
                50727 => { // AnalogBalance
                    metadata.analog_balance = Some(self.read_float_array(typ, count, val_offset, &val_bytes));
                }
                50728 => { // AsShotNeutral
                    metadata.as_shot_neutral = Some(self.read_float_array(typ, count, val_offset, &val_bytes));
                }
                50730 => { // BaselineExposure (0xC62A)
                    if typ == 5 || typ == 10 {
                        if let Some(val) = self.read_rational(val_offset) {
                            metadata.baseline_exposure = Some(val);
                        }
                    }
                }
                50734 => { // LinearResponseLimit
                    if typ == 5 || typ == 10 {
                        if let Some(val) = self.read_rational(val_offset) {
                            metadata.linear_response_limit = Some(val);
                        }
                    }
                }
                50739 => { // ShadowScale
                    if typ == 5 || typ == 10 {
                        if let Some(val) = self.read_rational(val_offset) {
                            metadata.shadow_scale = Some(val);
                        }
                    }
                }
                50936 => { // ProfileName
                    if typ == 2 {
                        metadata.profile_name = self.read_ascii(count, val_offset, &val_bytes);
                    }
                }
                50940 => { // ProfileToneCurve
                    metadata.profile_tone_curve = Some(self.read_float_array(typ, count, val_offset, &val_bytes));
                }
                50736 => { // LensInfo
                    metadata.lens_info = Some(self.read_float_array(typ, count, val_offset, &val_bytes));
                }
                50735 => { // CameraSerialNumber
                    if typ == 2 {
                        metadata.camera_serial_number = self.read_ascii(count, val_offset, &val_bytes);
                    }
                }
                51008 => { // OpcodeList1
                    metadata.opcode_list1 = Some(self.read_bytes(count, val_offset, &val_bytes));
                }
                51009 => { // OpcodeList2
                    metadata.opcode_list2 = Some(self.read_bytes(count, val_offset, &val_bytes));
                }
                51022 => { // OpcodeList3
                    metadata.opcode_list3 = Some(self.read_bytes(count, val_offset, &val_bytes));
                }
                51041 => { // NoiseProfile
                    if typ == 12 {
                        metadata.noise_profile = Some(self.read_double_array(typ, count, val_offset, &val_bytes));
                    }
                }
                _ => {}
            }

            entry_offset += 12;
        }

        // Parse next IFD
        if entry_offset + 4 <= self.data.len() {
            let next_ifd_offset = Self::read_u32_at(self.data, entry_offset, self.is_little_endian) as usize;
            self.parse_ifd(next_ifd_offset, metadata, visited);
        }
    }

    fn read_offsets(&self, typ: u16, count: usize, val_offset: usize, val_bytes: &[u8; 4]) -> Vec<usize> {
        let mut offsets = Vec::new();
        let type_size = match typ {
            3 => 2, // SHORT
            4 => 4, // LONG
            _ => return offsets,
        };
        let total_size = count * type_size;
        if total_size <= 4 {
            for i in 0..count {
                let off = i * type_size;
                let val = if type_size == 2 {
                    if self.is_little_endian {
                        u16::from_le_bytes([val_bytes[off], val_bytes[off+1]]) as usize
                    } else {
                        u16::from_be_bytes([val_bytes[off], val_bytes[off+1]]) as usize
                    }
                } else {
                    if self.is_little_endian {
                        u32::from_le_bytes([val_bytes[off], val_bytes[off+1], val_bytes[off+2], val_bytes[off+3]]) as usize
                    } else {
                        u32::from_be_bytes([val_bytes[off], val_bytes[off+1], val_bytes[off+2], val_bytes[off+3]]) as usize
                    }
                };
                offsets.push(val);
            }
        } else {
            if val_offset + total_size > self.data.len() {
                return offsets;
            }
            for i in 0..count {
                let off = val_offset + i * type_size;
                let val = if type_size == 2 {
                    Self::read_u16_at(self.data, off, self.is_little_endian) as usize
                } else {
                    Self::read_u32_at(self.data, off, self.is_little_endian) as usize
                };
                offsets.push(val);
            }
        }
        offsets
    }

    fn read_rational(&self, offset: usize) -> Option<f32> {
        if offset + 8 > self.data.len() {
            return None;
        }
        let num = Self::read_u32_at(self.data, offset, self.is_little_endian);
        let den = Self::read_u32_at(self.data, offset + 4, self.is_little_endian);
        if den == 0 {
            None
        } else {
            let s_num = num as i32;
            let s_den = den as i32;
            Some(s_num as f32 / s_den as f32)
        }
    }

    fn read_bytes(&self, count: usize, val_offset: usize, val_bytes: &[u8; 4]) -> Vec<u8> {
        if count <= 4 {
            val_bytes[0..count].to_vec()
        } else {
            if val_offset + count > self.data.len() {
                return Vec::new();
            }
            self.data[val_offset..val_offset + count].to_vec()
        }
    }

    fn read_ascii(&self, count: usize, val_offset: usize, val_bytes: &[u8; 4]) -> Option<String> {
        let bytes = if count <= 4 {
            val_bytes[0..count].to_vec()
        } else {
            if val_offset + count > self.data.len() {
                return None;
            }
            self.data[val_offset..val_offset + count].to_vec()
        };
        let trimmed_bytes = if let Some(pos) = bytes.iter().position(|&x| x == 0) {
            &bytes[0..pos]
        } else {
            &bytes[..]
        };
        String::from_utf8(trimmed_bytes.to_vec()).ok()
    }

    fn read_float_array(&self, typ: u16, count: usize, val_offset: usize, val_bytes: &[u8; 4]) -> Vec<f32> {
        let mut result = Vec::with_capacity(count);
        let type_size = match typ {
            3 => 2,
            4 => 4,
            5 | 10 => 8,
            11 => 4,
            _ => return result,
        };
        let total_size = count * type_size;
        
        let raw_bytes = if total_size <= 4 {
            val_bytes[0..total_size].to_vec()
        } else {
            if val_offset + total_size > self.data.len() {
                return result;
            }
            self.data[val_offset..val_offset + total_size].to_vec()
        };

        for i in 0..count {
            let offset = i * type_size;
            let val = match typ {
                3 => {
                    let v = if self.is_little_endian {
                        u16::from_le_bytes([raw_bytes[offset], raw_bytes[offset+1]])
                    } else {
                        u16::from_be_bytes([raw_bytes[offset], raw_bytes[offset+1]])
                    };
                    v as f32
                }
                4 => {
                    let v = if self.is_little_endian {
                        u32::from_le_bytes([raw_bytes[offset], raw_bytes[offset+1], raw_bytes[offset+2], raw_bytes[offset+3]])
                    } else {
                        u32::from_be_bytes([raw_bytes[offset], raw_bytes[offset+1], raw_bytes[offset+2], raw_bytes[offset+3]])
                    };
                    v as f32
                }
                5 => {
                    let num = if self.is_little_endian {
                        u32::from_le_bytes([raw_bytes[offset], raw_bytes[offset+1], raw_bytes[offset+2], raw_bytes[offset+3]])
                    } else {
                        u32::from_be_bytes([raw_bytes[offset], raw_bytes[offset+1], raw_bytes[offset+2], raw_bytes[offset+3]])
                    };
                    let den = if self.is_little_endian {
                        u32::from_le_bytes([raw_bytes[offset+4], raw_bytes[offset+5], raw_bytes[offset+6], raw_bytes[offset+7]])
                    } else {
                        u32::from_be_bytes([raw_bytes[offset+4], raw_bytes[offset+5], raw_bytes[offset+6], raw_bytes[offset+7]])
                    };
                    if den == 0 { 0.0 } else { num as f32 / den as f32 }
                }
                10 => {
                    let num = if self.is_little_endian {
                        i32::from_le_bytes([raw_bytes[offset], raw_bytes[offset+1], raw_bytes[offset+2], raw_bytes[offset+3]])
                    } else {
                        i32::from_be_bytes([raw_bytes[offset], raw_bytes[offset+1], raw_bytes[offset+2], raw_bytes[offset+3]])
                    };
                    let den = if self.is_little_endian {
                        i32::from_le_bytes([raw_bytes[offset+4], raw_bytes[offset+5], raw_bytes[offset+6], raw_bytes[offset+7]])
                    } else {
                        i32::from_be_bytes([raw_bytes[offset+4], raw_bytes[offset+5], raw_bytes[offset+6], raw_bytes[offset+7]])
                    };
                    if den == 0 { 0.0 } else { num as f32 / den as f32 }
                }
                11 => {
                    if self.is_little_endian {
                        f32::from_le_bytes([raw_bytes[offset], raw_bytes[offset+1], raw_bytes[offset+2], raw_bytes[offset+3]])
                    } else {
                        f32::from_be_bytes([raw_bytes[offset], raw_bytes[offset+1], raw_bytes[offset+2], raw_bytes[offset+3]])
                    }
                }
                _ => 0.0,
            };
            result.push(val);
        }
        result
    }

    fn read_double_array(&self, typ: u16, count: usize, val_offset: usize, _val_bytes: &[u8; 4]) -> Vec<f64> {
        let mut result = Vec::with_capacity(count);
        let type_size = match typ {
            12 => 8,
            _ => return result,
        };
        let total_size = count * type_size;
        if val_offset + total_size > self.data.len() {
            return result;
        }
        let raw_bytes = &self.data[val_offset..val_offset + total_size];
        for i in 0..count {
            let offset = i * type_size;
            let val = if self.is_little_endian {
                f64::from_le_bytes([
                    raw_bytes[offset], raw_bytes[offset+1], raw_bytes[offset+2], raw_bytes[offset+3],
                    raw_bytes[offset+4], raw_bytes[offset+5], raw_bytes[offset+6], raw_bytes[offset+7],
                ])
            } else {
                f64::from_be_bytes([
                    raw_bytes[offset], raw_bytes[offset+1], raw_bytes[offset+2], raw_bytes[offset+3],
                    raw_bytes[offset+4], raw_bytes[offset+5], raw_bytes[offset+6], raw_bytes[offset+7],
                ])
            };
            result.push(val);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_dng_metadata_parser() {
        let file_path = "../test_data/IMG_5851.DNG";
        let data = fs::read(file_path).expect("failed to read test DNG file");
        let parser = DngMetadataParser::new(&data).expect("failed to create DngMetadataParser");
        let metadata = parser.parse();
        println!("Parsed metadata: {:#?}", metadata);
        // Check that unique camera model and dng version are correctly read
        assert!(metadata.unique_camera_model.is_some());
        assert_eq!(metadata.unique_camera_model.as_deref(), Some("iPhone18,2 back camera"));
        assert!(metadata.dng_version.is_some());
        assert_eq!(metadata.dng_version, Some([1, 3, 0, 0]));
    }

    #[test]
    fn test_vignette_correction() {
        let file_path = "../test_data/IMG_5851.DNG";
        let data = fs::read(file_path).expect("failed to read test DNG file");
        let image = crate::extern_pipeline::get_raw_img_internal(&data);
        
        let demosaic_alg = pichromatic::demosaic::demosaic_algorithms::Amaze::default();
        let mut image = image.demosaic(demosaic_alg);
        
        let index_corner = 0; // top-left corner
        let val_before = image.rgb_data[index_corner];
        
        image.vignette(1.0);
        
        let val_after = image.rgb_data[index_corner];
        println!("Corner pixel before: {:?}, after: {:?}", val_before, val_after);
        assert!(val_after[0] > val_before[0]);
        assert!(val_after[1] > val_before[1]);
        assert!(val_after[2] > val_before[2]);
    }
}
