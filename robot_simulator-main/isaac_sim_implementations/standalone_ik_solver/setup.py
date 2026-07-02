from setuptools import setup


setup(
    name="standalone_ik_solver",
    version="0.1.0",
    packages=["standalone_ik_solver"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/standalone_ik_solver"]),
        ("share/standalone_ik_solver", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "scipy"],
    zip_safe=True,
    maintainer="wxl",
    maintainer_email="wxl@example.com",
    description="Non-Isaac IK solver for the dual-arm robot",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "standalone_ik_solver = standalone_ik_solver.__main__:main",
        ],
    },
)
