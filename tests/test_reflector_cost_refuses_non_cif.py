"""``GET /api/indexing/hough/reflector-cost`` names a file that is not a CIF instead of parsing it.

logs/orienta.log 2026-10-27: an .sht master reached the route after a method switch, diffpy
read the binary file as a CIF ("UnicodeDecodeError ... 0xa0", "StarError: bad encoding") and
the log got two tracebacks for one request.
"""
import asyncio
import logging

import pytest
from fastapi import HTTPException


def test_an_sht_master_is_refused_by_name_without_a_traceback(tmp_path, caplog):
    from backend.api.routes import indexing as route
    sht = tmp_path / "Al (Al) [cF4] {20kV}.sht"
    sht.write_bytes(b"\x00\x01binary master\xa0\xff" * 10)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(HTTPException) as err:
            asyncio.run(route.hough_reflector_cost(str(sht)))
    assert err.value.status_code == 400
    assert "not a CIF" in err.value.detail
    assert sht.name in err.value.detail
    assert not [r for r in caplog.records if r.exc_info]
