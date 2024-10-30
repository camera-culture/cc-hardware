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

#include "tmf882x_calib.h"
#include "tmf8828_shim.h"

const PROGMEM uint8_t tmf882x_calib_long_0[] = {
 0x19, 0x5, 0xBC, 0x0, 0x1, 0x1, 0x1, 0x0,
 0x1, 0xFF, 0xA0, 0xF, 0x7E, 0x56, 0x34, 0x12,
 0xCB, 0x90, 0x15, 0xBA, 0x1D, 0xE5, 0x7, 0x0,
 0x34, 0xBA, 0x7, 0x0, 0x73, 0x99, 0x7, 0x0,
 0x66, 0x97, 0x7, 0x0, 0x67, 0xA2, 0x7, 0x0,
 0xF, 0xA0, 0x7, 0x0, 0xF7, 0xBD, 0x7, 0x0,
 0x76, 0xBB, 0x7, 0x0, 0x93, 0xA3, 0x7, 0x0,
 0x94, 0x9B, 0x7, 0x0, 0xC2, 0xCE, 0x8, 0x0,
 0x29, 0x1D, 0x0, 0x0, 0x7D, 0xE, 0x0, 0x0,
 0x66, 0x10, 0x0, 0x0, 0xC4, 0x25, 0x0, 0x0,
 0xA8, 0x17, 0x0, 0x0, 0xB4, 0xE, 0x0, 0x0,
 0xE8, 0x1B, 0x0, 0x0, 0x84, 0xF, 0x0, 0x0,
 0x8A, 0x11, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x7, 0x0,
 0x0, 0x0, 0x7, 0x0, 0x0, 0x0, 0x7, 0x0,
 0x0, 0x0, 0x7, 0x0, 0x0, 0x0, 0x7, 0x0,
 0x0, 0x0, 0x7, 0x0, 0x0, 0x0, 0x7, 0x0,
 0x0, 0x0, 0x7, 0x0, 0x0, 0x0, 0x7, 0x0,
 0x0, 0x0, 0x7, 0x0, 0xA0, 0xF, 0x0, 0x0,
 0xA0, 0xF, 0x0, 0x0, 0xA0, 0xF, 0x0, 0x0,
 0xA0, 0xF, 0x0, 0x0, 0xA0, 0xF, 0x0, 0x0,
 0xA0, 0xF, 0x0, 0x0, 0xA0, 0xF, 0x0, 0x0,
 0xA0, 0xF, 0x0, 0x0, 0xA0, 0xF, 0x0, 0x0,
 0xA0, 0xF, 0x0, 0x0, 0x0, 0x0, 0xCA, 0x75,
};

const PROGMEM uint8_t tmf882x_calib_long_1[] = {
 0x19, 0x9, 0xBC, 0x0, 0x1, 0x2, 0x2, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x7F, 0x56, 0x34, 0x12,
 0xA5, 0x28, 0x78, 0x7F, 0x1F, 0xE6, 0x7, 0x0,
 0x76, 0xBB, 0x7, 0x0, 0x6D, 0x98, 0x7, 0x0,
 0xAF, 0x9C, 0x7, 0x0, 0xBD, 0xA9, 0x7, 0x0,
 0xF, 0xA0, 0x7, 0x0, 0x73, 0xC0, 0x7, 0x0,
 0x35, 0xBF, 0x7, 0x0, 0xCD, 0xA4, 0x7, 0x0,
 0x1F, 0xE6, 0x7, 0x0, 0x9B, 0xA9, 0x8, 0x0,
 0x9, 0x15, 0x0, 0x0, 0xF8, 0x8, 0x0, 0x0,
 0x4D, 0x7, 0x0, 0x0, 0x40, 0xB, 0x0, 0x0,
 0x7C, 0x13, 0x0, 0x0, 0x1F, 0xB, 0x0, 0x0,
 0xC7, 0x6, 0x0, 0x0, 0x18, 0x8, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x7F, 0x56, 0x34, 0x12,
 0xA5, 0x8C, 0x1E, 0x1F, 0x1D, 0xE5, 0x7, 0x0,
 0xB9, 0xBC, 0x7, 0x0, 0x7A, 0x9A, 0x7, 0x0,
 0x77, 0x94, 0x7, 0x0, 0xE3, 0x9E, 0x7, 0x0,
 0x67, 0xA2, 0x7, 0x0, 0xF7, 0xBD, 0x7, 0x0,
 0xB0, 0xB7, 0x7, 0x0, 0xAF, 0x9C, 0x7, 0x0,
 0x1D, 0xE5, 0x7, 0x0, 0xE3, 0x95, 0x8, 0x0,
 0x79, 0x14, 0x0, 0x0, 0xF1, 0xB, 0x0, 0x0,
 0xA4, 0x6, 0x0, 0x0, 0x1, 0x8, 0x0, 0x0,
 0x5B, 0x13, 0x0, 0x0, 0x2E, 0x9, 0x0, 0x0,
 0x31, 0x8, 0x0, 0x0, 0x80, 0xC, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0xAF, 0x75,
};

const PROGMEM uint8_t tmf882x_calib_short_0[] = {
 0x19, 0x6, 0xBC, 0x0, 0x0, 0x1, 0x1, 0x0,
 0x1, 0xFF, 0xA0, 0xF, 0x7E, 0x56, 0x34, 0x12,
 0xCB, 0x90, 0x15, 0xBA, 0xFF, 0xE8, 0x7, 0x0,
 0xCA, 0x2E, 0x7, 0x0, 0x7C, 0xFC, 0x6, 0x0,
 0xD0, 0xF5, 0x6, 0x0, 0x5F, 0x16, 0x7, 0x0,
 0x86, 0xC, 0x7, 0x0, 0x1A, 0x64, 0x7, 0x0,
 0xCF, 0x4D, 0x7, 0x0, 0x2D, 0xE, 0x7, 0x0,
 0x72, 0xEE, 0x6, 0x0, 0x71, 0xDA, 0x2, 0x0,
 0x4, 0x8, 0x0, 0x0, 0x1A, 0x4, 0x0, 0x0,
 0x93, 0x4, 0x0, 0x0, 0x9E, 0xA, 0x0, 0x0,
 0x90, 0x6, 0x0, 0x0, 0x5, 0x4, 0x0, 0x0,
 0x1A, 0x8, 0x0, 0x0, 0x52, 0x4, 0x0, 0x0,
 0xFB, 0x4, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1D, 0x76,
};

const PROGMEM uint8_t tmf882x_calib_short_1[] = {
 0x19, 0xA, 0xBC, 0x0, 0x0, 0x2, 0x2, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x7F, 0x56, 0x34, 0x12,
 0xA5, 0x28, 0x78, 0x7F, 0xFF, 0xE8, 0x7, 0x0,
 0xBD, 0x29, 0x7, 0x0, 0x4C, 0xFF, 0x6, 0x0,
 0x4C, 0xFF, 0x6, 0x0, 0x77, 0x14, 0x7, 0x0,
 0x0, 0x0, 0x7, 0x0, 0x1D, 0x65, 0x7, 0x0,
 0x0, 0x56, 0x7, 0x0, 0x1, 0xF, 0x7, 0x0,
 0xFF, 0xE8, 0x7, 0x0, 0x33, 0xD9, 0x2, 0x0,
 0xE, 0x6, 0x0, 0x0, 0xB4, 0x2, 0x0, 0x0,
 0x32, 0x2, 0x0, 0x0, 0x54, 0x3, 0x0, 0x0,
 0xB4, 0x5, 0x0, 0x0, 0x82, 0x3, 0x0, 0x0,
 0xF0, 0x1, 0x0, 0x0, 0x1D, 0x2, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x7F, 0x56, 0x34, 0x12,
 0xA5, 0x8C, 0x1E, 0x1F, 0x4D, 0xEF, 0x7, 0x0,
 0x98, 0x46, 0x7, 0x0, 0x6B, 0x15, 0x7, 0x0,
 0x7A, 0x1A, 0x7, 0x0, 0xE3, 0x1E, 0x7, 0x0,
 0xE1, 0x33, 0x7, 0x0, 0xD7, 0x5D, 0x7, 0x0,
 0xDD, 0x54, 0x7, 0x0, 0x34, 0xFD, 0x6, 0x0,
 0x4D, 0xEF, 0x7, 0x0, 0x20, 0xD0, 0x2, 0x0,
 0xFD, 0x5, 0x0, 0x0, 0x5C, 0x3, 0x0, 0x0,
 0xF1, 0x1, 0x0, 0x0, 0x3A, 0x2, 0x0, 0x0,
 0x75, 0x5, 0x0, 0x0, 0xB7, 0x2, 0x0, 0x0,
 0x52, 0x2, 0x0, 0x0, 0xCD, 0x3, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0xB, 0x76,
};

const PROGMEM uint8_t tmf882x_calib_short_1[] = {
 0x19, 0xA, 0xBC, 0x0, 0x0, 0x2, 0x2, 0x1,
 0x1, 0x1, 0xA0, 0xF, 0x7F, 0x56, 0x34, 0x12,
 0xA5, 0x28, 0x78, 0x7F, 0xFF, 0xE8, 0x7, 0x0,
 0xBD, 0x29, 0x7, 0x0, 0x4C, 0xFF, 0x6, 0x0,
 0x4C, 0xFF, 0x6, 0x0, 0x77, 0x14, 0x7, 0x0,
 0x0, 0x0, 0x7, 0x0, 0x1D, 0x65, 0x7, 0x0,
 0x0, 0x56, 0x7, 0x0, 0x1, 0xF, 0x7, 0x0,
 0xFF, 0xE8, 0x7, 0x0, 0x33, 0xD9, 0x2, 0x0,
 0xE, 0x6, 0x0, 0x0, 0xB4, 0x2, 0x0, 0x0,
 0x32, 0x2, 0x0, 0x0, 0x54, 0x3, 0x0, 0x0,
 0xB4, 0x5, 0x0, 0x0, 0x82, 0x3, 0x0, 0x0,
 0xF0, 0x1, 0x0, 0x0, 0x1D, 0x2, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x7F, 0x56, 0x34, 0x12,
 0xA5, 0x8C, 0x1E, 0x1F, 0x4D, 0xEF, 0x7, 0x0,
 0x98, 0x46, 0x7, 0x0, 0x6B, 0x15, 0x7, 0x0,
 0x7A, 0x1A, 0x7, 0x0, 0xE3, 0x1E, 0x7, 0x0,
 0xE1, 0x33, 0x7, 0x0, 0xD7, 0x5D, 0x7, 0x0,
 0xDD, 0x54, 0x7, 0x0, 0x34, 0xFD, 0x6, 0x0,
 0x4D, 0xEF, 0x7, 0x0, 0x20, 0xD0, 0x2, 0x0,
 0xFD, 0x5, 0x0, 0x0, 0x5C, 0x3, 0x0, 0x0,
 0xF1, 0x1, 0x0, 0x0, 0x3A, 0x2, 0x0, 0x0,
 0x75, 0x5, 0x0, 0x0, 0xB7, 0x2, 0x0, 0x0,
 0x52, 0x2, 0x0, 0x0, 0xCD, 0x3, 0x0, 0x0,
 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0xB, 0x76,
};