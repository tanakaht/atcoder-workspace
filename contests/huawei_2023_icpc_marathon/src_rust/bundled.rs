#[allow(dead_code)]
mod softfloat_wrapper {
    mod bf16 {
        use crate::softfloat_wrapper::{Float, RoundingMode, F128, F16, F32, F64};
        use softfloat_sys::{float16_t, float32_t};
        use std::borrow::Borrow;
        #[derive(Copy, Clone, Debug)]
        pub struct BF16(float16_t);
        impl BF16 {
            pub fn from_f32(v: f32) -> Self {
                F32::from_bits(v.to_bits()).to_bf16(RoundingMode::TiesToEven)
            }
            pub fn from_f64(v: f64) -> Self {
                F64::from_bits(v.to_bits()).to_bf16(RoundingMode::TiesToEven)
            }
        }
        fn to_f32(x: float16_t) -> float32_t {
            float32_t {
                v: (x.v as u32) << 16,
            }
        }
        fn from_f32(x: float32_t) -> float16_t {
            float16_t {
                v: (x.v >> 16) as u16,
            }
        }
        impl Float for BF16 {
            type Payload = u16;
            const EXPONENT_BIT: Self::Payload = 0xff;
            const FRACTION_BIT: Self::Payload = 0x7f;
            const SIGN_POS: usize = 15;
            const EXPONENT_POS: usize = 7;
            #[inline]
            fn set_payload(&mut self, x: Self::Payload) {
                self.0.v = x;
            }
            #[inline]
            fn from_bits(v: Self::Payload) -> Self {
                Self(float16_t { v })
            }
            #[inline]
            fn to_bits(&self) -> Self::Payload {
                self.0.v
            }
            #[inline]
            fn bits(&self) -> Self::Payload {
                self.to_bits()
            }
            fn add<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_add(to_f32(self.0), to_f32(x.borrow().0)) };
                Self(from_f32(ret))
            }
            fn sub<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_sub(to_f32(self.0), to_f32(x.borrow().0)) };
                Self(from_f32(ret))
            }
            fn mul<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_mul(to_f32(self.0), to_f32(x.borrow().0)) };
                Self(from_f32(ret))
            }
            fn fused_mul_add<T: Borrow<Self>>(&self, x: T, y: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe {
                    softfloat_sys::f32_mulAdd(to_f32(self.0), to_f32(x.borrow().0), to_f32(y.borrow().0))
                };
                Self(from_f32(ret))
            }
            fn div<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_div(to_f32(self.0), to_f32(x.borrow().0)) };
                Self(from_f32(ret))
            }
            fn rem<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_rem(to_f32(self.0), to_f32(x.borrow().0)) };
                Self(from_f32(ret))
            }
            fn sqrt(&self, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_sqrt(to_f32(self.0)) };
                Self(from_f32(ret))
            }
            fn eq<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f32_eq(to_f32(self.0), to_f32(x.borrow().0)) }
            }
            fn lt<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f32_lt(to_f32(self.0), to_f32(x.borrow().0)) }
            }
            fn le<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f32_le(to_f32(self.0), to_f32(x.borrow().0)) }
            }
            fn lt_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f32_lt_quiet(to_f32(self.0), to_f32(x.borrow().0)) }
            }
            fn le_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f32_le_quiet(to_f32(self.0), to_f32(x.borrow().0)) }
            }
            fn eq_signaling<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f32_eq_signaling(to_f32(self.0), to_f32(x.borrow().0)) }
            }
            fn is_signaling_nan(&self) -> bool {
                unsafe { softfloat_sys::f32_isSignalingNaN(to_f32(self.0)) }
            }
            fn from_u32(x: u32, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::ui32_to_f32(x) };
                Self(from_f32(ret))
            }
            fn from_u64(x: u64, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::ui64_to_f32(x) };
                Self(from_f32(ret))
            }
            fn from_i32(x: i32, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::i32_to_f32(x) };
                Self(from_f32(ret))
            }
            fn from_i64(x: i64, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::i64_to_f32(x) };
                Self(from_f32(ret))
            }
            fn to_u32(&self, rnd: RoundingMode, exact: bool) -> u32 {
                let ret = unsafe { softfloat_sys::f32_to_ui32(to_f32(self.0), rnd.to_softfloat(), exact) };
                ret as u32
            }
            fn to_u64(&self, rnd: RoundingMode, exact: bool) -> u64 {
                let ret = unsafe { softfloat_sys::f32_to_ui64(to_f32(self.0), rnd.to_softfloat(), exact) };
                ret
            }
            fn to_i32(&self, rnd: RoundingMode, exact: bool) -> i32 {
                let ret = unsafe { softfloat_sys::f32_to_i32(to_f32(self.0), rnd.to_softfloat(), exact) };
                ret as i32
            }
            fn to_i64(&self, rnd: RoundingMode, exact: bool) -> i64 {
                let ret = unsafe { softfloat_sys::f32_to_i64(to_f32(self.0), rnd.to_softfloat(), exact) };
                ret
            }
            fn to_f16(&self, rnd: RoundingMode) -> F16 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_to_f16(to_f32(self.0)) };
                F16::from_bits(ret.v)
            }
            fn to_bf16(&self, _rnd: RoundingMode) -> BF16 {
                BF16::from_bits(self.to_bits())
            }
            fn to_f32(&self, _rnd: RoundingMode) -> F32 {
                F32::from_bits(to_f32(self.0).v)
            }
            fn to_f64(&self, rnd: RoundingMode) -> F64 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_to_f64(to_f32(self.0)) };
                F64::from_bits(ret.v)
            }
            fn to_f128(&self, rnd: RoundingMode) -> F128 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_to_f128(to_f32(self.0)) };
                let mut v = 0u128;
                v |= ret.v[0] as u128;
                v |= (ret.v[1] as u128) << 64;
                F128::from_bits(v)
            }
            fn round_to_integral(&self, rnd: RoundingMode) -> Self {
                let ret =
                    unsafe { softfloat_sys::f32_roundToInt(to_f32(self.0), rnd.to_softfloat(), false) };
                Self(from_f32(ret))
            }
        }
    }
    mod f128 {
        use crate::softfloat_wrapper::{Float, RoundingMode, BF16, F16, F32, F64};
        use softfloat_sys::float128_t;
        use std::borrow::Borrow;
        #[derive(Copy, Clone, Debug)]
        pub struct F128(float128_t);
        impl F128 {
            pub fn from_f32(v: f32) -> Self {
                F32::from_bits(v.to_bits()).to_f128(RoundingMode::TiesToEven)
            }
            pub fn from_f64(v: f64) -> Self {
                F64::from_bits(v.to_bits()).to_f128(RoundingMode::TiesToEven)
            }
        }
        impl Float for F128 {
            type Payload = u128;
            const EXPONENT_BIT: Self::Payload = 0x7fff;
            const FRACTION_BIT: Self::Payload = 0xffff_ffff_ffff_ffff_ffff_ffff_ffff;
            const SIGN_POS: usize = 127;
            const EXPONENT_POS: usize = 112;
            #[inline]
            fn set_payload(&mut self, x: Self::Payload) {
                let x = [x as u64, (x >> 64) as u64];
                self.0.v = x;
            }
            #[inline]
            fn from_bits(v: Self::Payload) -> Self {
                let v = [v as u64, (v >> 64) as u64];
                Self(float128_t { v })
            }
            #[inline]
            fn to_bits(&self) -> Self::Payload {
                let mut ret = 0u128;
                ret |= self.0.v[0] as u128;
                ret |= (self.0.v[1] as u128) << 64;
                ret
            }
            #[inline]
            fn bits(&self) -> Self::Payload {
                self.to_bits()
            }
            fn add<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f128_add(self.0, x.borrow().0) };
                Self(ret)
            }
            fn sub<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f128_sub(self.0, x.borrow().0) };
                Self(ret)
            }
            fn mul<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f128_mul(self.0, x.borrow().0) };
                Self(ret)
            }
            fn fused_mul_add<T: Borrow<Self>>(&self, x: T, y: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f128_mulAdd(self.0, x.borrow().0, y.borrow().0) };
                Self(ret)
            }
            fn div<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f128_div(self.0, x.borrow().0) };
                Self(ret)
            }
            fn rem<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f128_rem(self.0, x.borrow().0) };
                Self(ret)
            }
            fn sqrt(&self, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f128_sqrt(self.0) };
                Self(ret)
            }
            fn eq<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f128_eq(self.0, x.borrow().0) }
            }
            fn lt<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f128_lt(self.0, x.borrow().0) }
            }
            fn le<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f128_le(self.0, x.borrow().0) }
            }
            fn lt_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f128_lt_quiet(self.0, x.borrow().0) }
            }
            fn le_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f128_le_quiet(self.0, x.borrow().0) }
            }
            fn eq_signaling<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f128_eq_signaling(self.0, x.borrow().0) }
            }
            fn is_signaling_nan(&self) -> bool {
                unsafe { softfloat_sys::f128_isSignalingNaN(self.0) }
            }
            fn from_u32(x: u32, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::ui32_to_f128(x) };
                Self(ret)
            }
            fn from_u64(x: u64, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::ui64_to_f128(x) };
                Self(ret)
            }
            fn from_i32(x: i32, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::i32_to_f128(x) };
                Self(ret)
            }
            fn from_i64(x: i64, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::i64_to_f128(x) };
                Self(ret)
            }
            fn to_u32(&self, rnd: RoundingMode, exact: bool) -> u32 {
                let ret = unsafe { softfloat_sys::f128_to_ui32(self.0, rnd.to_softfloat(), exact) };
                ret as u32
            }
            fn to_u64(&self, rnd: RoundingMode, exact: bool) -> u64 {
                let ret = unsafe { softfloat_sys::f128_to_ui64(self.0, rnd.to_softfloat(), exact) };
                ret
            }
            fn to_i32(&self, rnd: RoundingMode, exact: bool) -> i32 {
                let ret = unsafe { softfloat_sys::f128_to_i32(self.0, rnd.to_softfloat(), exact) };
                ret as i32
            }
            fn to_i64(&self, rnd: RoundingMode, exact: bool) -> i64 {
                let ret = unsafe { softfloat_sys::f128_to_i64(self.0, rnd.to_softfloat(), exact) };
                ret
            }
            fn to_f16(&self, rnd: RoundingMode) -> F16 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f128_to_f16(self.0) };
                F16::from_bits(ret.v)
            }
            fn to_bf16(&self, rnd: RoundingMode) -> BF16 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f128_to_f32(self.0) };
                BF16::from_bits((ret.v >> 16) as u16)
            }
            fn to_f32(&self, rnd: RoundingMode) -> F32 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f128_to_f32(self.0) };
                F32::from_bits(ret.v)
            }
            fn to_f64(&self, rnd: RoundingMode) -> F64 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f128_to_f64(self.0) };
                F64::from_bits(ret.v)
            }
            fn to_f128(&self, _rnd: RoundingMode) -> F128 {
                Self::from_bits(self.to_bits())
            }
            fn round_to_integral(&self, rnd: RoundingMode) -> Self {
                let ret = unsafe { softfloat_sys::f128_roundToInt(self.0, rnd.to_softfloat(), false) };
                Self(ret)
            }
        }
    }
    mod f16 {
        use crate::softfloat_wrapper::{Float, RoundingMode, BF16, F128, F32, F64};
        use softfloat_sys::float16_t;
        use std::borrow::Borrow;
        #[derive(Copy, Clone, Debug)]
        pub struct F16(float16_t);
        impl F16 {
            pub fn from_f32(v: f32) -> Self {
                F32::from_bits(v.to_bits()).to_f16(RoundingMode::TiesToEven)
            }
            pub fn from_f64(v: f64) -> Self {
                F64::from_bits(v.to_bits()).to_f16(RoundingMode::TiesToEven)
            }
        }
        impl Float for F16 {
            type Payload = u16;
            const EXPONENT_BIT: Self::Payload = 0x1f;
            const FRACTION_BIT: Self::Payload = 0x3ff;
            const SIGN_POS: usize = 15;
            const EXPONENT_POS: usize = 10;
            #[inline]
            fn set_payload(&mut self, x: Self::Payload) {
                self.0.v = x;
            }
            #[inline]
            fn from_bits(v: Self::Payload) -> Self {
                Self(float16_t { v })
            }
            #[inline]
            fn to_bits(&self) -> Self::Payload {
                self.0.v
            }
            #[inline]
            fn bits(&self) -> Self::Payload {
                self.to_bits()
            }
            fn add<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f16_add(self.0, x.borrow().0) };
                Self(ret)
            }
            fn sub<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f16_sub(self.0, x.borrow().0) };
                Self(ret)
            }
            fn mul<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f16_mul(self.0, x.borrow().0) };
                Self(ret)
            }
            fn fused_mul_add<T: Borrow<Self>>(&self, x: T, y: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f16_mulAdd(self.0, x.borrow().0, y.borrow().0) };
                Self(ret)
            }
            fn div<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f16_div(self.0, x.borrow().0) };
                Self(ret)
            }
            fn rem<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f16_rem(self.0, x.borrow().0) };
                Self(ret)
            }
            fn sqrt(&self, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f16_sqrt(self.0) };
                Self(ret)
            }
            fn eq<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f16_eq(self.0, x.borrow().0) }
            }
            fn lt<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f16_lt(self.0, x.borrow().0) }
            }
            fn le<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f16_le(self.0, x.borrow().0) }
            }
            fn lt_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f16_lt_quiet(self.0, x.borrow().0) }
            }
            fn le_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f16_le_quiet(self.0, x.borrow().0) }
            }
            fn eq_signaling<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f16_eq_signaling(self.0, x.borrow().0) }
            }
            fn is_signaling_nan(&self) -> bool {
                unsafe { softfloat_sys::f16_isSignalingNaN(self.0) }
            }
            fn from_u32(x: u32, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::ui32_to_f16(x) };
                Self(ret)
            }
            fn from_u64(x: u64, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::ui64_to_f16(x) };
                Self(ret)
            }
            fn from_i32(x: i32, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::i32_to_f16(x) };
                Self(ret)
            }
            fn from_i64(x: i64, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::i64_to_f16(x) };
                Self(ret)
            }
            fn to_u32(&self, rnd: RoundingMode, exact: bool) -> u32 {
                let ret = unsafe { softfloat_sys::f16_to_ui32(self.0, rnd.to_softfloat(), exact) };
                ret as u32
            }
            fn to_u64(&self, rnd: RoundingMode, exact: bool) -> u64 {
                let ret = unsafe { softfloat_sys::f16_to_ui64(self.0, rnd.to_softfloat(), exact) };
                ret
            }
            fn to_i32(&self, rnd: RoundingMode, exact: bool) -> i32 {
                let ret = unsafe { softfloat_sys::f16_to_i32(self.0, rnd.to_softfloat(), exact) };
                ret as i32
            }
            fn to_i64(&self, rnd: RoundingMode, exact: bool) -> i64 {
                let ret = unsafe { softfloat_sys::f16_to_i64(self.0, rnd.to_softfloat(), exact) };
                ret
            }
            fn to_f16(&self, _rnd: RoundingMode) -> F16 {
                Self::from_bits(self.to_bits())
            }
            fn to_bf16(&self, rnd: RoundingMode) -> BF16 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f16_to_f32(self.0) };
                BF16::from_bits((ret.v >> 16) as u16)
            }
            fn to_f32(&self, rnd: RoundingMode) -> F32 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f16_to_f32(self.0) };
                F32::from_bits(ret.v)
            }
            fn to_f64(&self, rnd: RoundingMode) -> F64 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f16_to_f64(self.0) };
                F64::from_bits(ret.v)
            }
            fn to_f128(&self, rnd: RoundingMode) -> F128 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f16_to_f128(self.0) };
                let mut v = 0u128;
                v |= ret.v[0] as u128;
                v |= (ret.v[1] as u128) << 64;
                F128::from_bits(v)
            }
            fn round_to_integral(&self, rnd: RoundingMode) -> Self {
                let ret = unsafe { softfloat_sys::f16_roundToInt(self.0, rnd.to_softfloat(), false) };
                Self(ret)
            }
        }
    }
    mod f32 {
        use crate::softfloat_wrapper::{Float, RoundingMode, BF16, F128, F16, F64};
        use softfloat_sys::float32_t;
        use std::borrow::Borrow;
        #[derive(Copy, Clone, Debug)]
        pub struct F32(float32_t);
        impl F32 {
            pub fn from_f32(v: f32) -> Self {
                Self::from_bits(v.to_bits())
            }
            pub fn from_f64(v: f64) -> Self {
                F64::from_bits(v.to_bits()).to_f32(RoundingMode::TiesToEven)
            }
        }
        impl Float for F32 {
            type Payload = u32;
            const EXPONENT_BIT: Self::Payload = 0xff;
            const FRACTION_BIT: Self::Payload = 0x7f_ffff;
            const SIGN_POS: usize = 31;
            const EXPONENT_POS: usize = 23;
            #[inline]
            fn set_payload(&mut self, x: Self::Payload) {
                self.0.v = x;
            }
            #[inline]
            fn from_bits(v: Self::Payload) -> Self {
                Self(float32_t { v })
            }
            #[inline]
            fn to_bits(&self) -> Self::Payload {
                self.0.v
            }
            #[inline]
            fn bits(&self) -> Self::Payload {
                self.to_bits()
            }
            fn add<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_add(self.0, x.borrow().0) };
                Self(ret)
            }
            fn sub<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_sub(self.0, x.borrow().0) };
                Self(ret)
            }
            fn mul<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_mul(self.0, x.borrow().0) };
                Self(ret)
            }
            fn fused_mul_add<T: Borrow<Self>>(&self, x: T, y: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_mulAdd(self.0, x.borrow().0, y.borrow().0) };
                Self(ret)
            }
            fn div<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_div(self.0, x.borrow().0) };
                Self(ret)
            }
            fn rem<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_rem(self.0, x.borrow().0) };
                Self(ret)
            }
            fn sqrt(&self, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_sqrt(self.0) };
                Self(ret)
            }
            fn eq<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f32_eq(self.0, x.borrow().0) }
            }
            fn lt<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f32_lt(self.0, x.borrow().0) }
            }
            fn le<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f32_le(self.0, x.borrow().0) }
            }
            fn lt_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f32_lt_quiet(self.0, x.borrow().0) }
            }
            fn le_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f32_le_quiet(self.0, x.borrow().0) }
            }
            fn eq_signaling<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f32_eq_signaling(self.0, x.borrow().0) }
            }
            fn is_signaling_nan(&self) -> bool {
                unsafe { softfloat_sys::f32_isSignalingNaN(self.0) }
            }
            fn from_u32(x: u32, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::ui32_to_f32(x) };
                Self(ret)
            }
            fn from_u64(x: u64, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::ui64_to_f32(x) };
                Self(ret)
            }
            fn from_i32(x: i32, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::i32_to_f32(x) };
                Self(ret)
            }
            fn from_i64(x: i64, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::i64_to_f32(x) };
                Self(ret)
            }
            fn to_u32(&self, rnd: RoundingMode, exact: bool) -> u32 {
                let ret = unsafe { softfloat_sys::f32_to_ui32(self.0, rnd.to_softfloat(), exact) };
                ret as u32
            }
            fn to_u64(&self, rnd: RoundingMode, exact: bool) -> u64 {
                let ret = unsafe { softfloat_sys::f32_to_ui64(self.0, rnd.to_softfloat(), exact) };
                ret
            }
            fn to_i32(&self, rnd: RoundingMode, exact: bool) -> i32 {
                let ret = unsafe { softfloat_sys::f32_to_i32(self.0, rnd.to_softfloat(), exact) };
                ret as i32
            }
            fn to_i64(&self, rnd: RoundingMode, exact: bool) -> i64 {
                let ret = unsafe { softfloat_sys::f32_to_i64(self.0, rnd.to_softfloat(), exact) };
                ret
            }
            fn to_f16(&self, rnd: RoundingMode) -> F16 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_to_f16(self.0) };
                F16::from_bits(ret.v)
            }
            fn to_bf16(&self, _rnd: RoundingMode) -> BF16 {
                BF16::from_bits((self.to_bits() >> 16) as u16)
            }
            fn to_f32(&self, _rnd: RoundingMode) -> F32 {
                Self::from_bits(self.to_bits())
            }
            fn to_f64(&self, rnd: RoundingMode) -> F64 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_to_f64(self.0) };
                F64::from_bits(ret.v)
            }
            fn to_f128(&self, rnd: RoundingMode) -> F128 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f32_to_f128(self.0) };
                let mut v = 0u128;
                v |= ret.v[0] as u128;
                v |= (ret.v[1] as u128) << 64;
                F128::from_bits(v)
            }
            fn round_to_integral(&self, rnd: RoundingMode) -> Self {
                let ret = unsafe { softfloat_sys::f32_roundToInt(self.0, rnd.to_softfloat(), false) };
                Self(ret)
            }
        }
    }
    mod f64 {
        use crate::softfloat_wrapper::{Float, RoundingMode, BF16, F128, F16, F32};
        use softfloat_sys::float64_t;
        use std::borrow::Borrow;
        #[derive(Copy, Clone, Debug)]
        pub struct F64(float64_t);
        impl F64 {
            pub fn from_f32(v: f32) -> Self {
                F32::from_bits(v.to_bits()).to_f64(RoundingMode::TiesToEven)
            }
            pub fn from_f64(v: f64) -> Self {
                Self::from_bits(v.to_bits())
            }
        }
        impl Float for F64 {
            type Payload = u64;
            const EXPONENT_BIT: Self::Payload = 0x7ff;
            const FRACTION_BIT: Self::Payload = 0xf_ffff_ffff_ffff;
            const SIGN_POS: usize = 63;
            const EXPONENT_POS: usize = 52;
            #[inline]
            fn set_payload(&mut self, x: Self::Payload) {
                self.0.v = x;
            }
            #[inline]
            fn from_bits(v: Self::Payload) -> Self {
                Self(float64_t { v })
            }
            #[inline]
            fn to_bits(&self) -> Self::Payload {
                self.0.v
            }
            #[inline]
            fn bits(&self) -> Self::Payload {
                self.to_bits()
            }
            fn add<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f64_add(self.0, x.borrow().0) };
                Self(ret)
            }
            fn sub<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f64_sub(self.0, x.borrow().0) };
                Self(ret)
            }
            fn mul<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f64_mul(self.0, x.borrow().0) };
                Self(ret)
            }
            fn fused_mul_add<T: Borrow<Self>>(&self, x: T, y: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f64_mulAdd(self.0, x.borrow().0, y.borrow().0) };
                Self(ret)
            }
            fn div<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f64_div(self.0, x.borrow().0) };
                Self(ret)
            }
            fn rem<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f64_rem(self.0, x.borrow().0) };
                Self(ret)
            }
            fn sqrt(&self, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::f64_sqrt(self.0) };
                Self(ret)
            }
            fn eq<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f64_eq(self.0, x.borrow().0) }
            }
            fn lt<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f64_lt(self.0, x.borrow().0) }
            }
            fn le<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f64_le(self.0, x.borrow().0) }
            }
            fn lt_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f64_lt_quiet(self.0, x.borrow().0) }
            }
            fn le_quiet<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f64_le_quiet(self.0, x.borrow().0) }
            }
            fn eq_signaling<T: Borrow<Self>>(&self, x: T) -> bool {
                unsafe { softfloat_sys::f64_eq_signaling(self.0, x.borrow().0) }
            }
            fn is_signaling_nan(&self) -> bool {
                unsafe { softfloat_sys::f64_isSignalingNaN(self.0) }
            }
            fn from_u32(x: u32, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::ui32_to_f64(x) };
                Self(ret)
            }
            fn from_u64(x: u64, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::ui64_to_f64(x) };
                Self(ret)
            }
            fn from_i32(x: i32, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::i32_to_f64(x) };
                Self(ret)
            }
            fn from_i64(x: i64, rnd: RoundingMode) -> Self {
                rnd.set();
                let ret = unsafe { softfloat_sys::i64_to_f64(x) };
                Self(ret)
            }
            fn to_u32(&self, rnd: RoundingMode, exact: bool) -> u32 {
                let ret = unsafe { softfloat_sys::f64_to_ui32(self.0, rnd.to_softfloat(), exact) };
                ret as u32
            }
            fn to_u64(&self, rnd: RoundingMode, exact: bool) -> u64 {
                let ret = unsafe { softfloat_sys::f64_to_ui64(self.0, rnd.to_softfloat(), exact) };
                ret
            }
            fn to_i32(&self, rnd: RoundingMode, exact: bool) -> i32 {
                let ret = unsafe { softfloat_sys::f64_to_i32(self.0, rnd.to_softfloat(), exact) };
                ret as i32
            }
            fn to_i64(&self, rnd: RoundingMode, exact: bool) -> i64 {
                let ret = unsafe { softfloat_sys::f64_to_i64(self.0, rnd.to_softfloat(), exact) };
                ret
            }
            fn to_f16(&self, rnd: RoundingMode) -> F16 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f64_to_f16(self.0) };
                F16::from_bits(ret.v)
            }
            fn to_bf16(&self, rnd: RoundingMode) -> BF16 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f64_to_f32(self.0) };
                BF16::from_bits((ret.v >> 16) as u16)
            }
            fn to_f32(&self, rnd: RoundingMode) -> F32 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f64_to_f32(self.0) };
                F32::from_bits(ret.v)
            }
            fn to_f64(&self, _rnd: RoundingMode) -> F64 {
                Self::from_bits(self.to_bits())
            }
            fn to_f128(&self, rnd: RoundingMode) -> F128 {
                rnd.set();
                let ret = unsafe { softfloat_sys::f64_to_f128(self.0) };
                let mut v = 0u128;
                v |= ret.v[0] as u128;
                v |= (ret.v[1] as u128) << 64;
                F128::from_bits(v)
            }
            fn round_to_integral(&self, rnd: RoundingMode) -> Self {
                let ret = unsafe { softfloat_sys::f64_roundToInt(self.0, rnd.to_softfloat(), false) };
                Self(ret)
            }
        }
    }
    pub use crate::softfloat_wrapper::bf16::BF16;
    pub use crate::softfloat_wrapper::f128::F128;
    pub use crate::softfloat_wrapper::f16::F16;
    pub use crate::softfloat_wrapper::f32::F32;
    pub use crate::softfloat_wrapper::f64::F64;
    use num_traits::{
        identities::{One, Zero},
        PrimInt,
    };
    use std::borrow::Borrow;
    use std::cmp::Ordering;
    use std::fmt::{LowerHex, UpperHex};
    #[derive(Copy, Clone, Debug)]
    pub enum RoundingMode {
        TiesToEven,
        TowardZero,
        TowardNegative,
        TowardPositive,
        TiesToAway,
    }
    impl RoundingMode {
        fn set(&self) {
            unsafe {
                softfloat_sys::softfloat_roundingMode_write_helper(self.to_softfloat());
            }
        }
        fn to_softfloat(&self) -> u8 {
            match self {
                RoundingMode::TiesToEven => softfloat_sys::softfloat_round_near_even,
                RoundingMode::TowardZero => softfloat_sys::softfloat_round_minMag,
                RoundingMode::TowardNegative => softfloat_sys::softfloat_round_min,
                RoundingMode::TowardPositive => softfloat_sys::softfloat_round_max,
                RoundingMode::TiesToAway => softfloat_sys::softfloat_round_near_maxMag,
            }
        }
    }
    #[derive(Copy, Clone, Debug, Default)]
    pub struct ExceptionFlags(u8);
    impl ExceptionFlags {
        const FLAG_INEXACT: u8 = softfloat_sys::softfloat_flag_inexact;
        const FLAG_INFINITE: u8 = softfloat_sys::softfloat_flag_infinite;
        const FLAG_INVALID: u8 = softfloat_sys::softfloat_flag_invalid;
        const FLAG_OVERFLOW: u8 = softfloat_sys::softfloat_flag_overflow;
        const FLAG_UNDERFLOW: u8 = softfloat_sys::softfloat_flag_underflow;
        pub fn from_bits(x: u8) -> Self {
            Self(x)
        }
        pub fn to_bits(&self) -> u8 {
            self.0
        }
        #[deprecated(since = "0.3.0", note = "Please use to_bits instead")]
        pub fn bits(&self) -> u8 {
            self.to_bits()
        }
        pub fn is_inexact(&self) -> bool {
            self.0 & Self::FLAG_INEXACT != 0
        }
        pub fn is_infinite(&self) -> bool {
            self.0 & Self::FLAG_INFINITE != 0
        }
        pub fn is_invalid(&self) -> bool {
            self.0 & Self::FLAG_INVALID != 0
        }
        pub fn is_overflow(&self) -> bool {
            self.0 & Self::FLAG_OVERFLOW != 0
        }
        pub fn is_underflow(&self) -> bool {
            self.0 & Self::FLAG_UNDERFLOW != 0
        }
        pub fn set(&self) {
            unsafe {
                softfloat_sys::softfloat_exceptionFlags_write_helper(self.to_bits());
            }
        }
        pub fn get(&mut self) {
            let x = unsafe { softfloat_sys::softfloat_exceptionFlags_read_helper() };
            self.0 = x;
        }
    }
    pub trait Float {
        type Payload: PrimInt + UpperHex + LowerHex;
        const EXPONENT_BIT: Self::Payload;
        const FRACTION_BIT: Self::Payload;
        const SIGN_POS: usize;
        const EXPONENT_POS: usize;
        fn set_payload(&mut self, x: Self::Payload);
        fn from_bits(v: Self::Payload) -> Self;
        fn to_bits(&self) -> Self::Payload;
        #[deprecated(since = "0.3.0", note = "Please use to_bits instead")]
        fn bits(&self) -> Self::Payload;
        fn add<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self;
        fn sub<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self;
        fn mul<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self;
        fn fused_mul_add<T: Borrow<Self>>(&self, x: T, y: T, rnd: RoundingMode) -> Self;
        fn div<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self;
        fn rem<T: Borrow<Self>>(&self, x: T, rnd: RoundingMode) -> Self;
        fn sqrt(&self, rnd: RoundingMode) -> Self;
        fn eq<T: Borrow<Self>>(&self, x: T) -> bool;
        fn lt<T: Borrow<Self>>(&self, x: T) -> bool;
        fn le<T: Borrow<Self>>(&self, x: T) -> bool;
        fn lt_quiet<T: Borrow<Self>>(&self, x: T) -> bool;
        fn le_quiet<T: Borrow<Self>>(&self, x: T) -> bool;
        fn eq_signaling<T: Borrow<Self>>(&self, x: T) -> bool;
        fn is_signaling_nan(&self) -> bool;
        fn from_u32(x: u32, rnd: RoundingMode) -> Self;
        fn from_u64(x: u64, rnd: RoundingMode) -> Self;
        fn from_i32(x: i32, rnd: RoundingMode) -> Self;
        fn from_i64(x: i64, rnd: RoundingMode) -> Self;
        fn to_u32(&self, rnd: RoundingMode, exact: bool) -> u32;
        fn to_u64(&self, rnd: RoundingMode, exact: bool) -> u64;
        fn to_i32(&self, rnd: RoundingMode, exact: bool) -> i32;
        fn to_i64(&self, rnd: RoundingMode, exact: bool) -> i64;
        fn to_f16(&self, rnd: RoundingMode) -> F16;
        fn to_bf16(&self, rnd: RoundingMode) -> BF16;
        fn to_f32(&self, rnd: RoundingMode) -> F32;
        fn to_f64(&self, rnd: RoundingMode) -> F64;
        fn to_f128(&self, rnd: RoundingMode) -> F128;
        fn round_to_integral(&self, rnd: RoundingMode) -> Self;
        #[inline]
        fn compare<T: Borrow<Self>>(&self, x: T) -> Option<Ordering> {
            let eq = self.eq(x.borrow());
            let lt = self.lt(x.borrow());
            if self.is_nan() || x.borrow().is_nan() {
                None
            } else if eq {
                Some(Ordering::Equal)
            } else if lt {
                Some(Ordering::Less)
            } else {
                Some(Ordering::Greater)
            }
        }
        #[inline]
        fn from_u8(x: u8, rnd: RoundingMode) -> Self
        where
            Self: Sized,
        {
            Self::from_u32(x as u32, rnd)
        }
        #[inline]
        fn from_u16(x: u16, rnd: RoundingMode) -> Self
        where
            Self: Sized,
        {
            Self::from_u32(x as u32, rnd)
        }
        #[inline]
        fn from_i8(x: i8, rnd: RoundingMode) -> Self
        where
            Self: Sized,
        {
            Self::from_i32(x as i32, rnd)
        }
        #[inline]
        fn from_i16(x: i16, rnd: RoundingMode) -> Self
        where
            Self: Sized,
        {
            Self::from_i32(x as i32, rnd)
        }
        #[inline]
        fn neg(&self) -> Self
        where
            Self: Sized,
        {
            let mut ret = Self::from_bits(self.to_bits());
            ret.set_sign(!self.sign());
            ret
        }
        #[inline]
        fn abs(&self) -> Self
        where
            Self: Sized,
        {
            let mut ret = Self::from_bits(self.to_bits());
            ret.set_sign(Self::Payload::zero());
            ret
        }
        #[inline]
        fn sign(&self) -> Self::Payload {
            (self.to_bits() >> Self::SIGN_POS) & Self::Payload::one()
        }
        #[inline]
        fn exponent(&self) -> Self::Payload {
            (self.to_bits() >> Self::EXPONENT_POS) & Self::EXPONENT_BIT
        }
        #[inline]
        fn fraction(&self) -> Self::Payload {
            self.to_bits() & Self::FRACTION_BIT
        }
        #[inline]
        fn is_positive(&self) -> bool {
            self.sign() == Self::Payload::zero()
        }
        #[inline]
        fn is_positive_zero(&self) -> bool {
            self.is_positive()
                && self.exponent() == Self::Payload::zero()
                && self.fraction() == Self::Payload::zero()
        }
        #[inline]
        fn is_positive_subnormal(&self) -> bool {
            self.is_positive()
                && self.exponent() == Self::Payload::zero()
                && self.fraction() != Self::Payload::zero()
        }
        #[inline]
        fn is_positive_normal(&self) -> bool {
            self.is_positive()
                && self.exponent() != Self::Payload::zero()
                && self.exponent() != Self::EXPONENT_BIT
        }
        #[inline]
        fn is_positive_infinity(&self) -> bool {
            self.is_positive()
                && self.exponent() == Self::EXPONENT_BIT
                && self.fraction() == Self::Payload::zero()
        }
        #[inline]
        fn is_negative(&self) -> bool {
            self.sign() == Self::Payload::one()
        }
        #[inline]
        fn is_negative_zero(&self) -> bool {
            self.is_negative()
                && self.exponent() == Self::Payload::zero()
                && self.fraction() == Self::Payload::zero()
        }
        #[inline]
        fn is_negative_subnormal(&self) -> bool {
            self.is_negative()
                && self.exponent() == Self::Payload::zero()
                && self.fraction() != Self::Payload::zero()
        }
        #[inline]
        fn is_negative_normal(&self) -> bool {
            self.is_negative()
                && self.exponent() != Self::Payload::zero()
                && self.exponent() != Self::EXPONENT_BIT
        }
        #[inline]
        fn is_negative_infinity(&self) -> bool {
            self.is_negative()
                && self.exponent() == Self::EXPONENT_BIT
                && self.fraction() == Self::Payload::zero()
        }
        #[inline]
        fn is_nan(&self) -> bool {
            self.exponent() == Self::EXPONENT_BIT && self.fraction() != Self::Payload::zero()
        }
        #[inline]
        fn is_zero(&self) -> bool {
            self.is_positive_zero() || self.is_negative_zero()
        }
        #[inline]
        fn is_subnormal(&self) -> bool {
            self.exponent() == Self::Payload::zero()
        }
        #[inline]
        fn set_sign(&mut self, x: Self::Payload) {
            self.set_payload(
                (self.to_bits() & !(Self::Payload::one() << Self::SIGN_POS))
                    | ((x & Self::Payload::one()) << Self::SIGN_POS),
            );
        }
        #[inline]
        fn set_exponent(&mut self, x: Self::Payload) {
            self.set_payload(
                (self.to_bits() & !(Self::EXPONENT_BIT << Self::EXPONENT_POS))
                    | ((x & Self::EXPONENT_BIT) << Self::EXPONENT_POS),
            );
        }
        #[inline]
        fn set_fraction(&mut self, x: Self::Payload) {
            self.set_payload((self.to_bits() & !Self::FRACTION_BIT) | (x & Self::FRACTION_BIT));
        }
        #[inline]
        fn positive_infinity() -> Self
        where
            Self: Sized,
        {
            let mut x = Self::from_bits(Self::Payload::zero());
            x.set_exponent(Self::EXPONENT_BIT);
            x
        }
        #[inline]
        fn positive_zero() -> Self
        where
            Self: Sized,
        {
            let x = Self::from_bits(Self::Payload::zero());
            x
        }
        #[inline]
        fn negative_infinity() -> Self
        where
            Self: Sized,
        {
            let mut x = Self::from_bits(Self::Payload::zero());
            x.set_sign(Self::Payload::one());
            x.set_exponent(Self::EXPONENT_BIT);
            x
        }
        #[inline]
        fn negative_zero() -> Self
        where
            Self: Sized,
        {
            let mut x = Self::from_bits(Self::Payload::zero());
            x.set_sign(Self::Payload::one());
            x
        }
        #[inline]
        fn quiet_nan() -> Self
        where
            Self: Sized,
        {
            let mut x = Self::from_bits(Self::Payload::zero());
            x.set_exponent(Self::EXPONENT_BIT);
            x.set_fraction(Self::Payload::one() << (Self::EXPONENT_POS - 1));
            x
        }
    }
}
// }}}
