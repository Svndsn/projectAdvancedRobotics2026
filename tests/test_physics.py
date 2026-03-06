import numpy as np

from utils.physics import amperes_law, biot_savarts_law


def test_biot_savart_2d():
    # Test the Biot-Savart law in 2D
    point = np.array([1.0, 0.0])
    I_wire = 10.0
    B = biot_savarts_law(point, I_wire)

    assert B.shape == (2,), "The magnetic field in 2D should be a 2D vector."
    assert not np.allclose(
        B, [0, 0]
    ), "Magnetic field should not be zero at this point."


def test_biot_savart_3d():
    # Test the Biot-Savart law in 3D
    point = np.array([1.0, 0.0, 0.0])
    I_wire = 10.0
    B = biot_savarts_law(point, I_wire)

    assert B.shape == (3,), "The magnetic field in 3D should be a 3D vector."
    assert not np.allclose(
        B, [0, 0, 0]
    ), "Magnetic field should not be zero at this point."


def test_amperes_law_2d():
    # Test Ampère's law with a square loop in 2D
    loop_points = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])  # Simple square loop
    I_wire = 10.0
    I_enc = amperes_law(loop_points, I_wire)

    assert isinstance(I_enc, float), "Enclosed current should be a scalar."
    assert I_enc != 0, "Enclosed current should not be zero."


def test_amperes_law_3d():
    # Test Ampère's law with a square loop in 3D
    loop_points = np.array(
        [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
    )  # Square loop in xy-plane
    I_wire = 10.0
    I_enc = amperes_law(loop_points, I_wire)

    assert isinstance(I_enc, float), "Enclosed current should be a scalar."
    assert I_enc != 0, "Enclosed current should not be zero."
