/*
 *****************************************************************************
 * Copyright by ams OSRAM AG * All rights are reserved. *
 *                                                                           *
 * IMPORTANT - PLEASE READ CAREFULLY BEFORE COPYING, INSTALLING OR USING     *
 * THE SOFTWARE.                                                             *
 *                                                                           *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       *
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         *
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS         *
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT  *
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,     *
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT          *
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,     *
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY     *
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT       *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE     *
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.      *
 *****************************************************************************
 */

// This is a calibration of my device. Replace it with one of yours.
// This calibration file must match the SPAD map ID. This is for
// my configs

#include "tmf8828_calib.h"
#include "tmf8828_shim.h"

const PROGMEM uint8_t tmf8828_calib_long_0[] = {
 0x19, 0x5, 0xBC, 0x0, 0x1, 0x4, 0x1, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x87, 0x56, 0x34, 0x12,
 0xAC, 0x49, 0x4D, 0x31, 0x18, 0xE3, 0x7, 0x0,
 0xF7, 0xBD, 0x7, 0x0, 0xAF, 0x9C, 0x7, 0x0,
 0x94, 0x9B, 0x7, 0x0, 0x67, 0xA2, 0x7, 0x0,
 0x94, 0x9B, 0x7, 0x0, 0xF7, 0xBD, 0x7, 0x0,
 0xB9, 0xBC, 0x7, 0x0, 0x7, 0xA6, 0x7, 0x0,
 0x18, 0xE3, 0x7, 0x0, 0x44, 0xD0, 0x8, 0x0,
 0xCA, 0x7, 0x0, 0x0, 0x9B, 0x4, 0x0, 0x0,
 0x52, 0xC, 0x0, 0x0, 0x11, 0x4, 0x0, 0x0,
 0x2D, 0xD, 0x0, 0x0, 0x5A, 0x5, 0x0, 0x0,
 0x1F, 0xC, 0x0, 0x0, 0xAF, 0x3, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x87, 0x56, 0x34, 0x12,
 0xA4, 0xA3, 0xE6, 0x81, 0x1A, 0xE4, 0x7, 0x0,
 0x73, 0xC0, 0x7, 0x0, 0xE3, 0x9E, 0x7, 0x0,
 0xC9, 0x9D, 0x7, 0x0, 0x3B, 0xA1, 0x7, 0x0,
 0xC9, 0x9D, 0x7, 0x0, 0x35, 0xBF, 0x7, 0x0,
 0xB9, 0xBC, 0x7, 0x0, 0x3B, 0xA1, 0x7, 0x0,
 0x1A, 0xE4, 0x7, 0x0, 0x43, 0xB5, 0x8, 0x0,
 0x2, 0x6, 0x0, 0x0, 0xC6, 0x4, 0x0, 0x0,
 0xD9, 0xA, 0x0, 0x0, 0xB9, 0x3, 0x0, 0x0,
 0xB, 0xA, 0x0, 0x0, 0x46, 0x4, 0x0, 0x0,
 0x7A, 0x7, 0x0, 0x0, 0xE0, 0x3, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0xB6, 0x75,
};
const PROGMEM uint8_t tmf8828_calib_long_1[] = {
 0x19, 0x6, 0xBC, 0x0, 0x1, 0x4, 0x1, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x87, 0x56, 0x34, 0x12,
 0x84, 0xB, 0x4C, 0xC4, 0x18, 0xE3, 0x7, 0x0,
 0x35, 0xBF, 0x7, 0x0, 0xC9, 0x9D, 0x7, 0x0,
 0xF, 0xA0, 0x7, 0x0, 0xCD, 0xA4, 0x7, 0x0,
 0xAF, 0x9C, 0x7, 0x0, 0xF7, 0xBD, 0x7, 0x0,
 0x76, 0xBB, 0x7, 0x0, 0x93, 0xA3, 0x7, 0x0,
 0x18, 0xE3, 0x7, 0x0, 0x8D, 0xD7, 0x8, 0x0,
 0x1, 0x5, 0x0, 0x0, 0x0, 0x5, 0x0, 0x0,
 0x33, 0x8, 0x0, 0x0, 0x9B, 0x4, 0x0, 0x0,
 0xB7, 0x9, 0x0, 0x0, 0x38, 0x4, 0x0, 0x0,
 0x39, 0x5, 0x0, 0x0, 0x42, 0x4, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x87, 0x56, 0x34, 0x12,
 0x24, 0xB3, 0xEE, 0x5, 0x15, 0xE2, 0x7, 0x0,
 0x76, 0xBB, 0x7, 0x0, 0x77, 0x94, 0x7, 0x0,
 0x7A, 0x9A, 0x7, 0x0, 0x7A, 0x9A, 0x7, 0x0,
 0x7A, 0x9A, 0x7, 0x0, 0xB0, 0xB7, 0x7, 0x0,
 0xB9, 0xBC, 0x7, 0x0, 0xC9, 0x9D, 0x7, 0x0,
 0x15, 0xE2, 0x7, 0x0, 0x60, 0xDB, 0x8, 0x0,
 0xAD, 0x4, 0x0, 0x0, 0x4D, 0x5, 0x0, 0x0,
 0x93, 0x6, 0x0, 0x0, 0x20, 0x6, 0x0, 0x0,
 0xEB, 0x8, 0x0, 0x0, 0xA8, 0x5, 0x0, 0x0,
 0xC, 0x4, 0x0, 0x0, 0x46, 0x5, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x32, 0x76,
};
const PROGMEM uint8_t tmf8828_calib_long_2[] = {
 0x19, 0x7, 0xBC, 0x0, 0x1, 0x4, 0x1, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x87, 0x56, 0x34, 0x12,
 0x98, 0xBE, 0x66, 0xAD, 0x5, 0xE1, 0x7, 0x0,
 0x76, 0xBB, 0x7, 0x0, 0x5F, 0x96, 0x7, 0x0,
 0x6D, 0x98, 0x7, 0x0, 0x7A, 0x9A, 0x7, 0x0,
 0x7A, 0x9A, 0x7, 0x0, 0x34, 0xBA, 0x7, 0x0,
 0x35, 0xBF, 0x7, 0x0, 0x7B, 0xA8, 0x7, 0x0,
 0x5, 0xE1, 0x7, 0x0, 0x9F, 0xD5, 0x8, 0x0,
 0x3D, 0xA, 0x0, 0x0, 0x3D, 0x4, 0x0, 0x0,
 0x68, 0xE, 0x0, 0x0, 0x7F, 0x5, 0x0, 0x0,
 0xE1, 0xE, 0x0, 0x0, 0x69, 0x4, 0x0, 0x0,
 0xBD, 0x8, 0x0, 0x0, 0xB7, 0x3, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x87, 0x56, 0x34, 0x12,
 0x90, 0x38, 0x48, 0xBC, 0x15, 0xE2, 0x7, 0x0,
 0x35, 0xBF, 0x7, 0x0, 0xE3, 0x9E, 0x7, 0x0,
 0x5F, 0x96, 0x7, 0x0, 0x66, 0x97, 0x7, 0x0,
 0xAF, 0x9C, 0x7, 0x0, 0xB9, 0xBC, 0x7, 0x0,
 0x35, 0xBF, 0x7, 0x0, 0x7, 0xA6, 0x7, 0x0,
 0x15, 0xE2, 0x7, 0x0, 0x59, 0xCB, 0x8, 0x0,
 0x3C, 0x8, 0x0, 0x0, 0x52, 0x4, 0x0, 0x0,
 0xD2, 0xC, 0x0, 0x0, 0x22, 0x4, 0x0, 0x0,
 0x84, 0x8, 0x0, 0x0, 0xEE, 0x3, 0x0, 0x0,
 0xD6, 0x6, 0x0, 0x0, 0x1A, 0x4, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2E, 0x76,
};
const PROGMEM uint8_t tmf8828_calib_long_3[] = {
 0x19, 0x8, 0xBC, 0x0, 0x1, 0x4, 0x1, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x87, 0x56, 0x34, 0x12,
 0x70, 0x20, 0xCE, 0xF7, 0x15, 0xE2, 0x7, 0x0,
 0x34, 0xBA, 0x7, 0x0, 0x5F, 0x96, 0x7, 0x0,
 0x66, 0x97, 0x7, 0x0, 0x66, 0x97, 0x7, 0x0,
 0xC9, 0x9D, 0x7, 0x0, 0x34, 0xBA, 0x7, 0x0,
 0xB9, 0xBC, 0x7, 0x0, 0x67, 0xA2, 0x7, 0x0,
 0x15, 0xE2, 0x7, 0x0, 0x50, 0xCA, 0x8, 0x0,
 0xFD, 0x5, 0x0, 0x0, 0xE8, 0x4, 0x0, 0x0,
 0x55, 0xB, 0x0, 0x0, 0x4C, 0x4, 0x0, 0x0,
 0x9F, 0x6, 0x0, 0x0, 0xCE, 0x3, 0x0, 0x0,
 0x35, 0x5, 0x0, 0x0, 0x8F, 0x4, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x87, 0x56, 0x34, 0x12,
 0x10, 0x48, 0x12, 0xBB, 0x15, 0xE2, 0x7, 0x0,
 0xB9, 0xBC, 0x7, 0x0, 0x5F, 0x96, 0x7, 0x0,
 0x5F, 0x96, 0x7, 0x0, 0x7A, 0x9A, 0x7, 0x0,
 0xC9, 0x9D, 0x7, 0x0, 0x6B, 0xB6, 0x7, 0x0,
 0x76, 0xBB, 0x7, 0x0, 0xF, 0xA0, 0x7, 0x0,
 0x15, 0xE2, 0x7, 0x0, 0x68, 0xC5, 0x8, 0x0,
 0x71, 0x4, 0x0, 0x0, 0x8, 0x6, 0x0, 0x0,
 0x6D, 0x8, 0x0, 0x0, 0xA7, 0x5, 0x0, 0x0,
 0x1D, 0x6, 0x0, 0x0, 0x6, 0x5, 0x0, 0x0,
 0x3F, 0x4, 0x0, 0x0, 0x1C, 0x5, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x50, 0x76,
};

const PROGMEM uint8_t tmf8828_calib_short_0[] = {
 0x19, 0x6, 0xBC, 0x0, 0x0, 0x4, 0x1, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x87, 0x56, 0x34, 0x12,
 0xAC, 0x49, 0x4D, 0x31, 0xF4, 0xE9, 0x7, 0x0,
 0x9E, 0x4C, 0x7, 0x0, 0x93, 0x23, 0x7, 0x0,
 0x66, 0x17, 0x7, 0x0, 0xCA, 0x2E, 0x7, 0x0,
 0xB2, 0xB, 0x7, 0x0, 0x48, 0x58, 0x7, 0x0,
 0xFF, 0x68, 0x7, 0x0, 0xF, 0x20, 0x7, 0x0,
 0xF4, 0xE9, 0x7, 0x0, 0xB, 0xDB, 0x2, 0x0,
 0x25, 0x2, 0x0, 0x0, 0x33, 0x1, 0x0, 0x0,
 0x6B, 0x3, 0x0, 0x0, 0x3B, 0x1, 0x0, 0x0,
 0xE7, 0x3, 0x0, 0x0, 0x7E, 0x1, 0x0, 0x0,
 0x7E, 0x3, 0x0, 0x0, 0xE0, 0x0, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x87, 0x56, 0x34, 0x12,
 0xA4, 0xA3, 0xE6, 0x81, 0x4D, 0xEF, 0x7, 0x0,
 0xCD, 0x47, 0x7, 0x0, 0xCD, 0x24, 0x7, 0x0,
 0xCD, 0x24, 0x7, 0x0, 0x7A, 0x1A, 0x7, 0x0,
 0x83, 0x13, 0x7, 0x0, 0xAE, 0x6C, 0x7, 0x0,
 0x4D, 0x6F, 0x7, 0x0, 0x84, 0x2D, 0x7, 0x0,
 0x4D, 0xEF, 0x7, 0x0, 0x35, 0xD1, 0x2, 0x0,
 0xA5, 0x1, 0x0, 0x0, 0x43, 0x1, 0x0, 0x0,
 0x0, 0x3, 0x0, 0x0, 0xE6, 0x0, 0x0, 0x0,
 0xBC, 0x2, 0x0, 0x0, 0xFB, 0x0, 0x0, 0x0,
 0x5B, 0x2, 0x0, 0x0, 0x9, 0x1, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0xF1, 0x75,
};
const PROGMEM uint8_t tmf8828_calib_short_1[] = {
 0x19, 0x7, 0xBC, 0x0, 0x0, 0x4, 0x1, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x87, 0x56, 0x34, 0x12,
 0x84, 0xB, 0x4C, 0xC4, 0xC5, 0xEB, 0x7, 0x0,
 0x6E, 0x4B, 0x7, 0x0, 0xAF, 0x1C, 0x7, 0x0,
 0xCD, 0x24, 0x7, 0x0, 0xB1, 0x41, 0x7, 0x0,
 0x79, 0x3, 0x7, 0x0, 0x14, 0x67, 0x7, 0x0,
 0x14, 0x67, 0x7, 0x0, 0x41, 0x27, 0x7, 0x0,
 0xC5, 0xEB, 0x7, 0x0, 0x3A, 0xD8, 0x2, 0x0,
 0x74, 0x1, 0x0, 0x0, 0x89, 0x1, 0x0, 0x0,
 0x56, 0x2, 0x0, 0x0, 0x39, 0x1, 0x0, 0x0,
 0xC4, 0x2, 0x0, 0x0, 0xED, 0x0, 0x0, 0x0,
 0xA9, 0x1, 0x0, 0x0, 0xF0, 0x0, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x87, 0x56, 0x34, 0x12,
 0x24, 0xB3, 0xEE, 0x5, 0x72, 0xEE, 0x7, 0x0,
 0x3, 0x49, 0x7, 0x0, 0x77, 0x14, 0x7, 0x0,
 0xBD, 0x29, 0x7, 0x0, 0x3B, 0x21, 0x7, 0x0,
 0x93, 0x8, 0x7, 0x0, 0xBC, 0x5C, 0x7, 0x0,
 0xB, 0x75, 0x7, 0x0, 0x7A, 0x1A, 0x7, 0x0,
 0x72, 0xEE, 0x7, 0x0, 0xC0, 0xD9, 0x2, 0x0,
 0x49, 0x1, 0x0, 0x0, 0x79, 0x1, 0x0, 0x0,
 0xAE, 0x1, 0x0, 0x0, 0xB8, 0x1, 0x0, 0x0,
 0x9D, 0x2, 0x0, 0x0, 0xA1, 0x1, 0x0, 0x0,
 0x4E, 0x1, 0x0, 0x0, 0x6D, 0x1, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x48, 0x76,
};
const PROGMEM uint8_t tmf8828_calib_short_2[] = {
 0x19, 0x8, 0xBC, 0x0, 0x0, 0x4, 0x1, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x87, 0x56, 0x34, 0x12,
 0x98, 0xBE, 0x66, 0xAD, 0xAE, 0xEC, 0x7, 0x0,
 0x98, 0x46, 0x7, 0x0, 0x56, 0x31, 0x7, 0x0,
 0x8F, 0x12, 0x7, 0x0, 0x1, 0xF, 0x7, 0x0,
 0xE4, 0xF, 0x7, 0x0, 0xA1, 0x5B, 0x7, 0x0,
 0x72, 0x6E, 0x7, 0x0, 0xB1, 0x41, 0x7, 0x0,
 0xAE, 0xEC, 0x7, 0x0, 0xF4, 0xDB, 0x2, 0x0,
 0xFC, 0x2, 0x0, 0x0, 0x13, 0x1, 0x0, 0x0,
 0x21, 0x4, 0x0, 0x0, 0x8C, 0x1, 0x0, 0x0,
 0x35, 0x4, 0x0, 0x0, 0x68, 0x1, 0x0, 0x0,
 0xAA, 0x2, 0x0, 0x0, 0x11, 0x1, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x87, 0x56, 0x34, 0x12,
 0x90, 0x38, 0x48, 0xBC, 0x72, 0xEE, 0x7, 0x0,
 0x98, 0x46, 0x7, 0x0, 0x93, 0x23, 0x7, 0x0,
 0x4C, 0xFF, 0x6, 0x0, 0x7B, 0x28, 0x7, 0x0,
 0x6D, 0x18, 0x7, 0x0, 0xA, 0x68, 0x7, 0x0,
 0x5C, 0x77, 0x7, 0x0, 0x56, 0x31, 0x7, 0x0,
 0x72, 0xEE, 0x7, 0x0, 0xFE, 0xDA, 0x2, 0x0,
 0x5A, 0x2, 0x0, 0x0, 0x2D, 0x1, 0x0, 0x0,
 0xBC, 0x3, 0x0, 0x0, 0x1A, 0x1, 0x0, 0x0,
 0x87, 0x2, 0x0, 0x0, 0xFD, 0x0, 0x0, 0x0,
 0x19, 0x2, 0x0, 0x0, 0x35, 0x1, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x54, 0x76,
};
const PROGMEM uint8_t tmf8828_calib_short_3[] = {
 0x19, 0x9, 0xBC, 0x0, 0x0, 0x4, 0x1, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x87, 0x56, 0x34, 0x12,
 0x70, 0x20, 0xCE, 0xF7, 0xC5, 0xEB, 0x7, 0x0,
 0xCD, 0x47, 0x7, 0x0, 0x66, 0x17, 0x7, 0x0,
 0xB2, 0xB, 0x7, 0x0, 0xE4, 0xF, 0x7, 0x0,
 0x59, 0xD, 0x7, 0x0, 0x5, 0x61, 0x7, 0x0,
 0x1F, 0x66, 0x7, 0x0, 0x7B, 0x28, 0x7, 0x0,
 0xC5, 0xEB, 0x7, 0x0, 0x18, 0x9E, 0x2, 0x0,
 0xD1, 0x1, 0x0, 0x0, 0x62, 0x1, 0x0, 0x0,
 0x4E, 0x3, 0x0, 0x0, 0x45, 0x1, 0x0, 0x0,
 0xE8, 0x1, 0x0, 0x0, 0x7, 0x1, 0x0, 0x0,
 0x89, 0x1, 0x0, 0x0, 0x33, 0x1, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x87, 0x56, 0x34, 0x12,
 0x10, 0x48, 0x12, 0xBB, 0x72, 0xEE, 0x7, 0x0,
 0x3, 0x49, 0x7, 0x0, 0x5B, 0x9, 0x7, 0x0,
 0x5B, 0x9, 0x7, 0x0, 0x17, 0x7, 0x7, 0x0,
 0x94, 0x1B, 0x7, 0x0, 0x5C, 0x51, 0x7, 0x0,
 0x1D, 0x65, 0x7, 0x0, 0xC9, 0x1D, 0x7, 0x0,
 0x72, 0xEE, 0x7, 0x0, 0x48, 0xA4, 0x2, 0x0,
 0x50, 0x1, 0x0, 0x0, 0xC2, 0x1, 0x0, 0x0,
 0x4D, 0x2, 0x0, 0x0, 0xA3, 0x1, 0x0, 0x0,
 0xD8, 0x1, 0x0, 0x0, 0x87, 0x1, 0x0, 0x0,
 0x52, 0x1, 0x0, 0x0, 0x79, 0x1, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x5C, 0x76,
};

const PROGMEM uint8_t tmf8828_calib_short_0[] = {
 0x19, 0x6, 0xBC, 0x0, 0x0, 0x4, 0x1, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x87, 0x56, 0x34, 0x12,
 0xAC, 0x49, 0x4D, 0x31, 0xF4, 0xE9, 0x7, 0x0,
 0x9E, 0x4C, 0x7, 0x0, 0x93, 0x23, 0x7, 0x0,
 0x66, 0x17, 0x7, 0x0, 0xCA, 0x2E, 0x7, 0x0,
 0xB2, 0xB, 0x7, 0x0, 0x48, 0x58, 0x7, 0x0,
 0xFF, 0x68, 0x7, 0x0, 0xF, 0x20, 0x7, 0x0,
 0xF4, 0xE9, 0x7, 0x0, 0xB, 0xDB, 0x2, 0x0,
 0x25, 0x2, 0x0, 0x0, 0x33, 0x1, 0x0, 0x0,
 0x6B, 0x3, 0x0, 0x0, 0x3B, 0x1, 0x0, 0x0,
 0xE7, 0x3, 0x0, 0x0, 0x7E, 0x1, 0x0, 0x0,
 0x7E, 0x3, 0x0, 0x0, 0xE0, 0x0, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x87, 0x56, 0x34, 0x12,
 0xA4, 0xA3, 0xE6, 0x81, 0x4D, 0xEF, 0x7, 0x0,
 0xCD, 0x47, 0x7, 0x0, 0xCD, 0x24, 0x7, 0x0,
 0xCD, 0x24, 0x7, 0x0, 0x7A, 0x1A, 0x7, 0x0,
 0x83, 0x13, 0x7, 0x0, 0xAE, 0x6C, 0x7, 0x0,
 0x4D, 0x6F, 0x7, 0x0, 0x84, 0x2D, 0x7, 0x0,
 0x4D, 0xEF, 0x7, 0x0, 0x35, 0xD1, 0x2, 0x0,
 0xA5, 0x1, 0x0, 0x0, 0x43, 0x1, 0x0, 0x0,
 0x0, 0x3, 0x0, 0x0, 0xE6, 0x0, 0x0, 0x0,
 0xBC, 0x2, 0x0, 0x0, 0xFB, 0x0, 0x0, 0x0,
 0x5B, 0x2, 0x0, 0x0, 0x9, 0x1, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0xF1, 0x75,
};
const PROGMEM uint8_t tmf8828_calib_short_1[] = {
 0x19, 0x7, 0xBC, 0x0, 0x0, 0x4, 0x1, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x87, 0x56, 0x34, 0x12,
 0x84, 0xB, 0x4C, 0xC4, 0xC5, 0xEB, 0x7, 0x0,
 0x6E, 0x4B, 0x7, 0x0, 0xAF, 0x1C, 0x7, 0x0,
 0xCD, 0x24, 0x7, 0x0, 0xB1, 0x41, 0x7, 0x0,
 0x79, 0x3, 0x7, 0x0, 0x14, 0x67, 0x7, 0x0,
 0x14, 0x67, 0x7, 0x0, 0x41, 0x27, 0x7, 0x0,
 0xC5, 0xEB, 0x7, 0x0, 0x3A, 0xD8, 0x2, 0x0,
 0x74, 0x1, 0x0, 0x0, 0x89, 0x1, 0x0, 0x0,
 0x56, 0x2, 0x0, 0x0, 0x39, 0x1, 0x0, 0x0,
 0xC4, 0x2, 0x0, 0x0, 0xED, 0x0, 0x0, 0x0,
 0xA9, 0x1, 0x0, 0x0, 0xF0, 0x0, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x87, 0x56, 0x34, 0x12,
 0x24, 0xB3, 0xEE, 0x5, 0x72, 0xEE, 0x7, 0x0,
 0x3, 0x49, 0x7, 0x0, 0x77, 0x14, 0x7, 0x0,
 0xBD, 0x29, 0x7, 0x0, 0x3B, 0x21, 0x7, 0x0,
 0x93, 0x8, 0x7, 0x0, 0xBC, 0x5C, 0x7, 0x0,
 0xB, 0x75, 0x7, 0x0, 0x7A, 0x1A, 0x7, 0x0,
 0x72, 0xEE, 0x7, 0x0, 0xC0, 0xD9, 0x2, 0x0,
 0x49, 0x1, 0x0, 0x0, 0x79, 0x1, 0x0, 0x0,
 0xAE, 0x1, 0x0, 0x0, 0xB8, 0x1, 0x0, 0x0,
 0x9D, 0x2, 0x0, 0x0, 0xA1, 0x1, 0x0, 0x0,
 0x4E, 0x1, 0x0, 0x0, 0x6D, 0x1, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x48, 0x76,
};
const PROGMEM uint8_t tmf8828_calib_short_2[] = {
 0x19, 0x8, 0xBC, 0x0, 0x0, 0x4, 0x1, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x87, 0x56, 0x34, 0x12,
 0x98, 0xBE, 0x66, 0xAD, 0xAE, 0xEC, 0x7, 0x0,
 0x98, 0x46, 0x7, 0x0, 0x56, 0x31, 0x7, 0x0,
 0x8F, 0x12, 0x7, 0x0, 0x1, 0xF, 0x7, 0x0,
 0xE4, 0xF, 0x7, 0x0, 0xA1, 0x5B, 0x7, 0x0,
 0x72, 0x6E, 0x7, 0x0, 0xB1, 0x41, 0x7, 0x0,
 0xAE, 0xEC, 0x7, 0x0, 0xF4, 0xDB, 0x2, 0x0,
 0xFC, 0x2, 0x0, 0x0, 0x13, 0x1, 0x0, 0x0,
 0x21, 0x4, 0x0, 0x0, 0x8C, 0x1, 0x0, 0x0,
 0x35, 0x4, 0x0, 0x0, 0x68, 0x1, 0x0, 0x0,
 0xAA, 0x2, 0x0, 0x0, 0x11, 0x1, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x87, 0x56, 0x34, 0x12,
 0x90, 0x38, 0x48, 0xBC, 0x72, 0xEE, 0x7, 0x0,
 0x98, 0x46, 0x7, 0x0, 0x93, 0x23, 0x7, 0x0,
 0x4C, 0xFF, 0x6, 0x0, 0x7B, 0x28, 0x7, 0x0,
 0x6D, 0x18, 0x7, 0x0, 0xA, 0x68, 0x7, 0x0,
 0x5C, 0x77, 0x7, 0x0, 0x56, 0x31, 0x7, 0x0,
 0x72, 0xEE, 0x7, 0x0, 0xFE, 0xDA, 0x2, 0x0,
 0x5A, 0x2, 0x0, 0x0, 0x2D, 0x1, 0x0, 0x0,
 0xBC, 0x3, 0x0, 0x0, 0x1A, 0x1, 0x0, 0x0,
 0x87, 0x2, 0x0, 0x0, 0xFD, 0x0, 0x0, 0x0,
 0x19, 0x2, 0x0, 0x0, 0x35, 0x1, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x54, 0x76,
};
const PROGMEM uint8_t tmf8828_calib_short_3[] = {
 0x19, 0x9, 0xBC, 0x0, 0x0, 0x4, 0x1, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x87, 0x56, 0x34, 0x12,
 0x70, 0x20, 0xCE, 0xF7, 0xC5, 0xEB, 0x7, 0x0,
 0xCD, 0x47, 0x7, 0x0, 0x66, 0x17, 0x7, 0x0,
 0xB2, 0xB, 0x7, 0x0, 0xE4, 0xF, 0x7, 0x0,
 0x59, 0xD, 0x7, 0x0, 0x5, 0x61, 0x7, 0x0,
 0x1F, 0x66, 0x7, 0x0, 0x7B, 0x28, 0x7, 0x0,
 0xC5, 0xEB, 0x7, 0x0, 0x18, 0x9E, 0x2, 0x0,
 0xD1, 0x1, 0x0, 0x0, 0x62, 0x1, 0x0, 0x0,
 0x4E, 0x3, 0x0, 0x0, 0x45, 0x1, 0x0, 0x0,
 0xE8, 0x1, 0x0, 0x0, 0x7, 0x1, 0x0, 0x0,
 0x89, 0x1, 0x0, 0x0, 0x33, 0x1, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x87, 0x56, 0x34, 0x12,
 0x10, 0x48, 0x12, 0xBB, 0x72, 0xEE, 0x7, 0x0,
 0x3, 0x49, 0x7, 0x0, 0x5B, 0x9, 0x7, 0x0,
 0x5B, 0x9, 0x7, 0x0, 0x17, 0x7, 0x7, 0x0,
 0x94, 0x1B, 0x7, 0x0, 0x5C, 0x51, 0x7, 0x0,
 0x1D, 0x65, 0x7, 0x0, 0xC9, 0x1D, 0x7, 0x0,
 0x72, 0xEE, 0x7, 0x0, 0x48, 0xA4, 0x2, 0x0,
 0x50, 0x1, 0x0, 0x0, 0xC2, 0x1, 0x0, 0x0,
 0x4D, 0x2, 0x0, 0x0, 0xA3, 0x1, 0x0, 0x0,
 0xD8, 0x1, 0x0, 0x0, 0x87, 0x1, 0x0, 0x0,
 0x52, 0x1, 0x0, 0x0, 0x79, 0x1, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x5C, 0x76,
};
