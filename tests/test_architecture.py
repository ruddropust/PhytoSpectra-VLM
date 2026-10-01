from phytospectra.model import PhytoCore


def test_phytocore_parameter_contract():
    assert PhytoCore().parameter_count() == 1_998_346

