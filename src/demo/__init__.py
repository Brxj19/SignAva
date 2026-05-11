__all__ = ["generate_demo_videos"]


def __getattr__(name: str):
    if name == "generate_demo_videos":
        from src.demo.generate_demo_videos import generate_demo_videos

        return generate_demo_videos
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
