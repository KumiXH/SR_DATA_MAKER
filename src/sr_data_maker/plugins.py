from __future__ import annotations

from sr_data_maker.core.registry import Registry
from sr_data_maker.generators.degradation import DegradationGenerator
from sr_data_maker.generators.teacher_sr import TeacherSRGenerator
from sr_data_maker.runners.degradation.classical import ClassicalDegradationRunner
from sr_data_maker.runners.teacher.codeformer import CodeFormerRunner
from sr_data_maker.runners.teacher.resshift import ResShiftRunner
from sr_data_maker.runners.teacher.gfpgan import GFPGANRunner
from sr_data_maker.runners.teacher.hat import HATAdapter
from sr_data_maker.runners.teacher.realesrgan import RealESRGANRunner
from sr_data_maker.runners.teacher.stablesr import StableSRRunner
from sr_data_maker.runners.teacher.supir import SUPIRRunner
from sr_data_maker.runners.teacher.swinir import SwinIRAdapter
from sr_data_maker.runners.teacher.vqfr import VQFRRunner
from sr_data_maker.sources.image_folder import ImageFolderSourceReader

SOURCE_READERS = Registry("source_readers")
GENERATORS = Registry("generators")
RUNNERS = Registry("runners")


def register_builtins() -> None:
    if not SOURCE_READERS._items:
        SOURCE_READERS.register("ImageFolderSourceReader")(ImageFolderSourceReader)
    if not GENERATORS._items:
        GENERATORS.register("DegradationGenerator")(DegradationGenerator)
        GENERATORS.register("TeacherSRGenerator")(TeacherSRGenerator)
    if not RUNNERS._items:
        RUNNERS.register("ClassicalDegradationRunner")(ClassicalDegradationRunner)
        RUNNERS.register("RealESRGANRunner")(RealESRGANRunner)
        RUNNERS.register("SwinIRAdapter")(SwinIRAdapter)
        RUNNERS.register("HATAdapter")(HATAdapter)
        RUNNERS.register("GFPGANRunner")(GFPGANRunner)
        RUNNERS.register("CodeFormerRunner")(CodeFormerRunner)
        RUNNERS.register("VQFRRunner")(VQFRRunner)
        RUNNERS.register("StableSRRunner")(StableSRRunner)
        RUNNERS.register("ResShiftRunner")(ResShiftRunner)
        RUNNERS.register("SUPIRRunner")(SUPIRRunner)
