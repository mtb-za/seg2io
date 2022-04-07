"""
Read SEG-2 files as xarray datatypes.

These were originally written by natstoik, and extended by Martin Bentley.

Author: Matt Hall
Email: matt@agilescientific.com

Licence: Apache 2.0

Copyright 2022 Agile Scientific

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from struct import unpack
import numpy as np
import xarray as xr

DATA_FORMAT_CODES = {
    1: np.int16,
    2: np.int32,
    # 3: 20-bit floating point,  # WTF no thanks.
    4: np.float32,
    5: np.float64,
}


def read_seg2_header(fname):

    file_header = {}
    with open(fname, 'rb') as f:

        file_descriptor_block = f.read(32)

        # Check file descriptor indicates SEG2.
        first_byte, = unpack(b'B', file_descriptor_block[0:1])
        if first_byte == 0x55:
            endian = b'<'
        elif first_byte == 0x3a:
            endian = b'>'
        else:
            print('This is not a SEG2 file.')

        # Check SEG2 revision number (only rev 1 exists).
        rev, = unpack(endian + b'H', file_descriptor_block[2:4])
        if rev != 1:
            print('This might not work, file is rev {}.'.format(rev))

        # Get size of trace pointer sub-block and number of traces.
        # Documentation uses M and N for these quantities.
        M, = unpack(endian + b'H', file_descriptor_block[4:6])
        N, = unpack(endian + b'H', file_descriptor_block[6:8])

        # Define the string and line terminators.
        (size_of_string_terminator,
         first_string_terminator_char,
         second_string_terminator_char,
         size_of_line_terminator,
         first_line_terminator_char,
         second_line_terminator_char
         ) = unpack(b'BccBcc', file_descriptor_block[8:14])

        # Assemble the string terminator.
        string_terminator = first_string_terminator_char
        if size_of_string_terminator == 2:
            string_terminator += second_string_terminator_char

        # Assemble the line terminator.
        line_terminator = first_line_terminator_char
        if size_of_line_terminator == 2:
            line_terminator += second_line_terminator_char

        # Read the trace pointer sub-block and retrieve all the pointers.
        trace_pointer_sub_block = f.read(M)
        trace_pointers = []
        for i in range(N):
            index = i * 4
            this_word = trace_pointer_sub_block[index:index + 4]
            pointer, = unpack(endian + b'L', this_word)
            trace_pointers.append(pointer)

        # Read the free format section, aka file header.
        while True:
            offset_word = f.read(2)
            offset, = unpack(endian + b'H', offset_word)
            if offset == 0:
                break
            this_string = f.read(offset-2)
            try:
                k, v = this_string.split()
            except:
                continue
            file_header[k.decode('utf-8')] = v[:-1].decode('utf-8')

    return file_header


def read_seg(fname):
    """
    Reads a SEG-2 file and extracts metadata and the data.

    Args:
        fname (str): The SEG-2 filename. Common extensions are .seg, .seg2,
            .dat, sg2, .rad

    Returns:
        tuple. (list, ndarray). The trace headers as a list (usually they
        are not interesting), and the data as an ndarray.
    """
    trace_headers = []
    traces = []

    with open(fname, 'rb') as f:

        file_descriptor_block = f.read(32)

        # Check file descriptor indicates SEG2.
        first_byte, = unpack(b'B', file_descriptor_block[0:1])
        if first_byte == 0x55:
            endian = b'<'
        elif first_byte == 0x3a:
            endian = b'>'
        else:
            print('This is not a SEG2 file.')

        # Check SEG2 revision number (only rev 1 exists).
        rev, = unpack(endian + b'H', file_descriptor_block[2:4])
        if rev != 1:
            print('This might not work, file is rev {}.'.format(rev))

        # Get size of trace pointer sub-block and number of traces.
        # Documentation uses M and N for these quantities.
        M, = unpack(endian + b'H', file_descriptor_block[4:6])
        N, = unpack(endian + b'H', file_descriptor_block[6:8])

        # Define the string and line terminators.
        (size_of_string_terminator,
         first_string_terminator_char,
         second_string_terminator_char,
         size_of_line_terminator,
         first_line_terminator_char,
         second_line_terminator_char
         ) = unpack(b'BccBcc', file_descriptor_block[8:14])

        # Assemble the string terminator.
        string_terminator = first_string_terminator_char
        if size_of_string_terminator == 2:
            string_terminator += second_string_terminator_char

        # Assemble the line terminator.
        line_terminator = first_line_terminator_char
        if size_of_line_terminator == 2:
            line_terminator += second_line_terminator_char

        # Read the trace pointer sub-block and retrieve all the pointers.
        trace_pointer_sub_block = f.read(M)
        trace_pointers = []
        for i in range(N):
            index = i * 4
            this_word = trace_pointer_sub_block[index:index + 4]
            pointer, = unpack(endian + b'L', this_word)
            trace_pointers.append(pointer)

        # Read traces
        for p in trace_pointers:

            f.seek(p, 0)
            trace_descriptor_block = f.read(32)

            # Read block ID and other parameters.
            ID, = unpack(endian + b'H', trace_descriptor_block[0:2])
            assert ID == 0x4422

            # We can read X, Y, NS; but we don't care about X, Y:
            _, _, NS = unpack(endian + b'HII', trace_descriptor_block[2:12])
            format_code, = unpack(endian + b'B', trace_descriptor_block[12:13])
            try:
                dtype = DATA_FORMAT_CODES[format_code]
                # print(f'dtype set to {dtype}')
            except KeyError:
                print("Data type not supported.")
            nbytes = dtype().nbytes

            # Read the free format section, aka trace header.
            trace_header = {}
            while True:
                offset_word = f.read(2)
                offset, = unpack(endian + b'H', offset_word)
                if offset == 0:
                    break
                this_string = f.read(offset-2)
                try:
                    k, v = this_string.split()
                except:
                    # Malformed; give up.
                    continue
                # Add the characters, skipping the string termination char.
                trace_header[k.decode('utf-8')] = v[:-1].decode('utf-8')
                trace_headers.append(trace_header)

            # There should be no bytes after the zero-offset indicator.
            # But I need to move on 1 byte to continue reading the data.
            # I don't know if it's '1' or 'size_of_string_terminator'.
            # I don't know if this is just a USRadar thing.
            f.seek(1, 1)

            # Read the trace data.
            raw = f.read(NS * nbytes)
            trace = np.fromstring(raw, dtype=dtype)
            traces.append(trace)

        x_arr = xr.DataArray(np.stack(traces))

        f.close()

    return trace_headers, x_arr
