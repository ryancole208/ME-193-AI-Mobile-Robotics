"""Look up webcams by name instead of by index (Windows/DirectShow only).

Windows can reorder camera indexes after a reboot or a USB re-plug, so a fixed
index can silently open the laptop camera instead of the overhead one. This
lists DirectShow video devices in the same order OpenCV's CAP_DSHOW uses, using
only ctypes (no extra packages).
"""
import ctypes
import platform
from ctypes import POINTER, byref, c_void_p, wintypes


def _guid(s):
    import uuid
    return (ctypes.c_byte * 16).from_buffer_copy(uuid.UUID(s).bytes_le)


CLSID_SystemDeviceEnum = "62BE5D10-60EB-11D0-BD3B-00A0C911CE86"
IID_ICreateDevEnum = "29840822-5B84-11D0-BD3B-00A0C911CE86"
CLSID_VideoInputDeviceCategory = "860BB310-5D01-11D0-BD3B-00A0C911CE86"
IID_IPropertyBag = "55272A00-42CB-11CE-8135-00AA004BB851"
CLSCTX_INPROC_SERVER = 1
COINIT_MULTITHREADED = 0
VT_BSTR = 8


class _VARIANT(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort),
                ("r2", ctypes.c_ushort), ("r3", ctypes.c_ushort),
                ("val", c_void_p), ("pad", c_void_p)]


def _method(obj, index, restype, *argtypes):
    """Fetch vtable entry `index` of COM object `obj` as a callable."""
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
    proto = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)
    fn = proto(vtbl[index])
    return lambda *a: fn(obj, *a)


def _release(obj):
    if obj:
        _method(obj, 2, ctypes.c_ulong)()


def list_camera_names():
    """Return DirectShow video device names; list position == OpenCV index."""
    if platform.system() != "Windows":
        return []
    ole32 = ctypes.windll.ole32
    oleaut32 = ctypes.windll.oleaut32
    hr = ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
    # S_OK/S_FALSE: we own an init and must undo it. Other codes (e.g. the thread
    # already uses another mode) still leave COM usable.
    owns_init = hr in (0, 1)
    names = []
    dev_enum = c_void_p()
    enum_mon = c_void_p()
    try:
        clsid, iid = _guid(CLSID_SystemDeviceEnum), _guid(IID_ICreateDevEnum)
        if ole32.CoCreateInstance(byref(clsid), None, CLSCTX_INPROC_SERVER,
                                  byref(iid), byref(dev_enum)) != 0:
            return []
        cat = _guid(CLSID_VideoInputDeviceCategory)
        create_enum = _method(dev_enum, 3, ctypes.c_long, c_void_p, POINTER(c_void_p),
                              wintypes.DWORD)
        # S_FALSE (1) means the category is empty.
        if create_enum(byref(cat), byref(enum_mon), 0) != 0 or not enum_mon:
            return []
        nxt = _method(enum_mon, 3, ctypes.c_long, wintypes.ULONG, POINTER(c_void_p),
                      POINTER(wintypes.ULONG))
        iid_bag = _guid(IID_IPropertyBag)
        while True:
            mon = c_void_p()
            fetched = wintypes.ULONG()
            if nxt(1, byref(mon), byref(fetched)) != 0 or not mon:
                break
            name = "?"
            bag = c_void_p()
            try:
                bind = _method(mon, 9, ctypes.c_long, c_void_p, c_void_p, c_void_p,
                               POINTER(c_void_p))
                if bind(None, None, byref(iid_bag), byref(bag)) == 0 and bag:
                    var = _VARIANT()
                    read = _method(bag, 3, ctypes.c_long, wintypes.LPCWSTR,
                                   POINTER(_VARIANT), c_void_p)
                    if read("FriendlyName", byref(var), None) == 0 and var.vt == VT_BSTR:
                        name = ctypes.wstring_at(var.val)
                    oleaut32.VariantClear(byref(var))
            finally:
                _release(bag)
                _release(mon)
            names.append(name)
    finally:
        _release(enum_mon)
        _release(dev_enum)
        if owns_init:
            ole32.CoUninitialize()
    return names


def find_camera_index(name):
    """Index of the first camera whose name contains `name` (case-insensitive), or None."""
    want = name.lower()
    for i, n in enumerate(list_camera_names()):
        if want in n.lower():
            return i
    return None


if __name__ == "__main__":
    for i, n in enumerate(list_camera_names()):
        print(i, n)
