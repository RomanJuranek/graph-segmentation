"""
Support for reading TESCAN XRay data to an numpy array

"""


import gzip
import logging
from pathlib import Path
from xml.dom import minidom

import cv2
import numpy as np
from skimage.morphology import dilation


class XrayHeaderError(Exception):
    """Error in the header of binary data"""
    pass


class IntegrityError(Exception):
    """Inconsistent image sizes or missing fields"""
    pass


class InvalidFieldError(Exception):
    """Missing files required by the reader"""
    pass


_tescan_magic = "Tescan X-ray map"  # Magic in xray.map
_bse_name = "bse.png"
_eds_names = ["xray.map", "xray.map.gz"]
_mask_name = "pixelmask.png"


def read_header_fields(buf):
    """Iterator through header fields until END is encoutered"""
    while True:
        s = buf.read(80)  # This is fixed
        key = s[:9].decode().strip()
        value = s[10:].decode().strip()
        if key in ["WIDTH","HEIGHT","NCHANNELS"]:
            value = int(value)
        if key == "END":
            break
        yield key, value


def read_header(buf):
    """Read xray.map header and return dict with keys and values"""
    magic = buf.read(len(_tescan_magic)).decode()
    if magic != _tescan_magic:
        raise XrayHeaderError
    header = dict(read_header_fields(buf))
    return header


def decode_plane(data, depth, W, dtype):
    """Decode pixels from data based on the given bit depth"""
    if depth == 0:
        pass

    elif depth == 1:
        pixels = np.unpackbits(data, axis=1)[:,:W]
        return pixels.astype(dtype)

    elif depth == 2:
        h,w = data.shape
        pixels = np.empty((h,4*w), dtype=dtype)
        for i in range(4):
            pixels[:,i::4] = np.bitwise_and(data, 0x03<<(2*i)) >> (2*i)
        return pixels[:,:W]

    elif depth == 4:
        h,w = data.shape
        pixels = np.empty((h,2*w), dtype=dtype)
        for i in range(2):
            pixels[:,i::2] = np.bitwise_and(data, 0x0F<<(4*i)) >> (4*i)
        return pixels[:,:W]

    elif depth == 8:
        return data.astype(dtype)

    elif depth == 16:
        h,w = data.shape
        pixels = np.empty((h,w//2), dtype=dtype)
        pixels[:] = data[:,0::2] + 256 * data[:,1::2].astype(dtype=dtype)
        return pixels

    else:
        raise ValueError(f"Cannot decode {depth}-bit data")


def read_data(buf, header):
    """Read the binary data to an array of shape (C,H,W)"""
    def full_row_width(w, bpp):
        """Get size of data row in bytes (from XRayMap.cpp)"""
        return ((w)*(bpp)+7)//8

    try:
        shape = N,H,W = header["NCHANNELS"],header["HEIGHT"],header["WIDTH"]
    except KeyError:
        raise IntegrityError("Header must contain at least NCHANNELS, WIDTH and HEIGHT information")

    depth = np.frombuffer(buf.read(N), "b")
    max_depth = np.max(depth)
    target_dtype = np.uint8
    if max_depth > 8:
        target_dtype = np.uint16
    elif max_depth > 16:
        target_dtype = np.uint32
    data = np.empty(shape, dtype=target_dtype)

    logging.debug(f"Target data shape is {data.shape}")

    for chn_id, channel_depth in enumerate(depth):
        logging.debug(f"Channel {chn_id}, depth {channel_depth}")
        if channel_depth == 0:
            data[chn_id,...] = 0
            continue
        row_width = full_row_width(W, channel_depth)
        logging.debug(f"Row width is {row_width}")
        channel_data = np.array([np.frombuffer(buf.read(row_width), dtype=np.uint8) for _ in range(H)]) # Reads data from file stored in bytes
        logging.debug(f"Channel data shape is {channel_data.shape}")
        data[chn_id] = decode_plane(channel_data, channel_depth, W, dtype=target_dtype)

    if H == 1:
        data = np.moveaxis(data.squeeze(), 0, -1)
    else:
        data = data.reshape(N,-1).T

    return data


def read_xray_map(filename):
    """Read EDS data from file"""
    f = gzip.open(filename,"rb") if filename.as_posix().endswith(".gz") else open(filename,"rb")
    eds_header = read_header(f)
    eds = read_data(f, eds_header)
    f.close()
    return eds

              
class Field:
    """
    Class for representation of BSE+EDS data

    New instance can be initialized uwing from_path() function
        field = Field.from_path(path_to_field)

    or using constructor Field() and passing BSE, EDS and locations of measurements.

    Note
    ----
    All fields, DOT and HIRES, are represented in a common structure. So there is no
    difference between them once they are loaded. While this is not optimal, it simplifies
    the algorithms acting upon the structure greatly. However, processing of HIRES fields
    is definitely not efficient and could be accelerated by customizing the algorithms
    for the dense data.

    Members
    -------
    bse : ndarray
        (H,W) array with BSE data.
    eds : ndarray
        (N,D) where N is the number of pixels with EDS measurement and
        D is the dimensionality. Usually N>10000 and D=3000
    r,c : ndarray
        Arrays of size N with rows and columns of EDS measurements.
    sample_distance : integent
        maximal distance between samples that is considered to be direct
        neighborhood.

    Properties
    ----------
    normalized_bse : ndarray
        BSE image with zero mean and unit standard deviation
    pixels : ndarray
        (N,2) array composed from r and c. Just for convenience.
    shape : tuple
        3-tuple (H,W,D)
    size : int
        Total number of pixels - H*W
    n_measurements : int
        Number of EDS measurements - N
    n_bands : int
        Number of bands in EDS spectrum - D

    Methods
    -------
    valid_pixel_array : ndarray
        (H,W) array where pixel (i,j) is set to 1 is it is contained in r,c - mask
        of valid pixels
    valid_mask : ndarray
        Mask of pixels with nearby EDS measurement - obtained by dilation of
        valid_pixel array with square element of size sample_distacne.


    TODO:
    * Some mathods can actually be transformed to properties

    """
    def __init__(self, bse, eds, r, c, sample_distance=5):
        self.bse = bse
        self.eds = eds
        self.r = r
        self.c = c
        self.sample_distance = sample_distance
        
        n_eds_pixels = eds.shape[0]
        if n_eds_pixels != self.r.size:
            raise InvalidFieldError(f"Number of pixels in mask ({self.r.size}) does not match number of EDS measurements ({n_eds_pixels})")

        if bse.ndim != 2:
            raise InvalidFieldError("BSE image must have 2 dimension")

    @property
    def normalized_bse(self):
        #arr = ndi.convolve(self.bse.astype("f"), np.array([[1,2,1],[2,4,2],[1,2,1]])/16)
        arr = self.bse.astype("f")
        return arr / 65536
    
    def valid_pixel_array(self):
        return np.squeeze(self.as_dense_array(np.ones(self.n_measurements,"u1")))
    
    def valid_mask(self):
        mask = self.valid_pixel_array()
        if self.sample_distance > 1:
            r = np.ones((self.sample_distance,)*2,"u1")
            mask = dilation(mask, selem=r)
        return mask
    
    @property
    def pixels(self):
        return np.array([self.c, self.r],"f").T

    def eds_array(self, band=None, aggregate=False):
        """ () """
        if isinstance(band, int):
            band = (band,band+1)
        slc = slice(*band) if band is not None else slice(None)
        X = self.eds[:,slc]
        if aggregate:
            X = X.sum(axis=-1,keepdims=True)
        return self.as_dense_array(X)

    def as_dense_array(self, X):
        if X.ndim == 1:
            X = X.reshape(-1,1)
        n_rows, n_dim = X.shape
        if n_rows != self.r.size:
            raise ValueError(f"X must have {self.r.size} rows, got {n_rows}")
        dst_shape = self.bse.shape[:2] + X.shape[1:2]
        dst = np.zeros(dst_shape, dtype=X.dtype)
        for r,c,x in zip(self.r, self.c, X):
            dst[r,c,...] = x
        return dst

    @staticmethod
    def from_path(data_path, min_count=800):

        def get_sample_distance():
            config_xml_file = data_path.parent / "configuration" / "profile.xml"
            doc = minidom.parse(config_xml_file.as_posix())
            ps = doc.getElementsByTagName("PixelSpacing")[0]
            pixel_spacing = int(ps.childNodes[0].data)
            ds = doc.getElementsByTagName("DotSpacing")[0]
            dot_spacing = int(ds.childNodes[0].data)
            return int(dot_spacing/pixel_spacing)

        data_path = Path(data_path)
        bse_file = data_path/_bse_name

        if not bse_file.exists():
            raise InvalidFieldError("Missing bse.png file")
        eds_file = next((data_path/x for x in _eds_names if (data_path/x).exists()), None)
        if not eds_file :
            raise InvalidFieldError("Missing xray.map file")
        mask_file = data_path/_mask_name

        bse = cv2.imread(bse_file.as_posix(), cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread(mask_file.as_posix(), cv2.IMREAD_GRAYSCALE) > 0 if mask_file.exists() else None
        eds = read_xray_map(eds_file)
        sample_distance = get_sample_distance()
        
        if mask is None:  # HIRES - generate full mask
            mask = np.ones_like(bse, "u1")
            sample_distance = 1
        
        r, c = np.nonzero(mask)

        valid_eds = eds.sum(axis=1) > min_count
        eds = eds[valid_eds]
        r = r[valid_eds]
        c = c[valid_eds]

        return Field(bse, eds, r, c, sample_distance)

    @property
    def shape(self):
        return self.bse.shape[:2] + self.eds.shape[1:2]
    
    @property
    def size(self):
        return np.prod(self.shape[:2])
    
    @property
    def n_measurements(self):
        return self.r.size
                
    @property
    def n_bands(self):
        return self.eds.shape[1]

