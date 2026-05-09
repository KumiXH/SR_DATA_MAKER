from PIL import Image

from sr_data_maker.runners.degradation.classical import ClassicalDegradationRunner


def test_classical_degradation_resizes_by_scale_and_records_params():
    runner = ClassicalDegradationRunner(
        scale=2,
        blur={"enabled": False},
        resize={"enabled": True, "mode": "bicubic"},
        noise={"enabled": False},
        jpeg={"enabled": False},
        seed=123,
    )
    image = Image.new("RGB", (8, 6), "white")

    output = runner.run({"image": image}, context=None)

    assert output.outputs["image"].size == (4, 3)
    assert output.meta["params"]["scale"] == 2
    assert output.meta["params"]["resize"]["enabled"] is True
