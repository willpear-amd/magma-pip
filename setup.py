from setuptools import Extension, find_packages, setup
from setuptools.command.sdist import sdist
from setuptools.command.build_ext import build_ext
import setuptools.command.install
from setuptools.dist import Distribution

import glob
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

ROOT_DIR = os.path.abspath('.')
PACKAGE_NAME = "magma"
MAGMA_INCLUDE = os.path.join(ROOT_DIR, "include")
MAGMA_LIB = os.path.join(ROOT_DIR, "lib")
MAGMA_VERSION = "2.6.1"

def detect_gpu_arch():
    if os.getenv("ROCM_PATH") != "" and os.path.exists(os.path.join(os.getenv("ROCM_PATH"), "bin", "hipcc")):
        return "rocm"
    elif os.getenv("CUDA_HOME") != "" and os.path.exists(os.path.join(os.getenv("ROCM_PATH"), "bin", "nvcc")):
        return "cuda"
    else:
        print("No CUDA or ROCm installation found. Building for CPU.")
        return None


def find_library(header):

    searching_for = f"Searching for {header}"

    for folder in MAGMA_INCLUDE:
        if (Path(folder) / header).exists():
            print(f"{searching_for} in {Path(folder) / header}. Found in Magma Include.")
            return True, None, None
        
    print(f"{searching_for}. Didn't find in Magma Include.")


def get_version():
    with open(ROOT_DIR / "magma/version.txt") as f:
        version = f.readline().strip()
    sha = "Unknown"

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR)).decode("ascii").strip()
    except Exception:
        pass

    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]

    return version, sha



def subprocess_run(command, cwd=None, env=None):
    """ This attempts to run a shell command. """
    print(f"Running: {command}")

    try:

        result = subprocess.check_output(
            command,
            cwd=ROOT_DIR,
            shell=True
        )
        return result
            
    except subprocess.CalledProcessError as e:
        print(f"Command Failed with exit code {e.returncode}:")
        print(e.stderr)
        raise
    
class BuildExtension(build_ext):
    """ A build extension class for the Magma Library. """
    def run(self):

        self.gpu_arch = detect_gpu_arch()

        if self.gpu_arch == "cuda":
            print("Building MAGMA for CUDA...")

        elif self.gpu_arch == "rocm":

            cpus = str(int(os.cpu_count()))

            mkl_root = '/opt/conda/envs/py_3.10'
            os.environ["MKLROOT"] = '/opt/conda/envs/py_3.10'
            subprocess.check_output('export PATH="${PATH}:/opt/rocm/bin"', shell=True)

            print("Building MAGMA for ROCm...")

            make_inc = os.path.join(ROOT_DIR,"make.inc")

            if os.path.exists(make_inc):
                os.remove(make_inc)

            # Copy the make.inc file to this directory
            shutil.copy2(os.path.join(ROOT_DIR,"make.inc-examples","make.inc.hip-gcc-mkl"), make_inc)

            gfx_arch =  os.environ.get("PYTORCH_ROCM_ARCH").split(';')

            if len(gfx_arch)==0:
                gfx_arch = subprocess.check_output(['bash', '-c', 'rocm_agent_enumerator']).decode('utf-8').split('\n') 
                
                # Remove empty entries and and gfx000
                gfx_arch = [gfx for gfx in gfx_arch if gfx not in ['gfx000', '']]


            # Writing gfx arch to file
            with open(os.path.join(make_inc), "a") as f:
                for gfx in gfx_arch:
                    f.write(f"\nDEVCCFLAGS += --offload-arch={gfx}")
                    
            # Build commands
            hip_build = f"/usr/bin/make     -f make.gen.hipMAGMA    -j {cpus}"
            so_build =  f"/usr/bin/make     lib/libmagma.so         -j {cpus} MKLROOT={mkl_root}"
            testing =   f"/usr/bin/make     testing/testing_dgemm   -j {cpus} MKLROOT={mkl_root}"
            
            try:
                print(subprocess_run(hip_build, cwd=ROOT_DIR))
                print("End of hip build")

            except Exception as e:
                raise RuntimeError(f"Error running MAGMA library build comand: {e}") from e

            os.environ["LANG"] = 'C.UTF-8'
            try:
                print(subprocess_run(so_build, cwd=ROOT_DIR))
                print("End of main build")
            except Exception as e:
                raise RuntimeError(f"Error running MAGMA library build comand: {e}") from e

                
            try:
                subprocess_run(testing, cwd=ROOT_DIR)
                print("End of testing build")

            except Exception as e:
                raise RuntimeError(f"Error running MAGMA library build comand: {e}") from e
            

        build_lib = self.build_lib

        print(f"Build destination = {build_lib}")
        
        # Magma folder in wheel file
        package_target = os.path.join(ROOT_DIR, build_lib, PACKAGE_NAME)

        # Target destination for libmagma.so
        target_lib = os.path.join(package_target, "lib")

        os.makedirs(target_lib, exist_ok=True)

        # Move libmagma.so to package dir
        shutil.copy(os.path.join(MAGMA_LIB, "libmagma.so"), os.path.join(target_lib, "libmagma.so"))
        shutil.copytree(MAGMA_INCLUDE, os.path.join(package_target, "include"))

        # Call parent
        super().run()


    def build_extension(self, ext):

        ext.libraries.append(PACKAGE_NAME)
        ext.include_dirs.append(MAGMA_INCLUDE)
        ext.library_dirs.append(MAGMA_LIB)
        
        super().build_extension(ext)

class clean(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import glob
        import re


        shutil.rmtree(os.path.join(ROOT_DIR, "magma.egg-info"))  if os.path.exists(os.path.join(ROOT_DIR, "magma.egg-info")) else 0
        shutil.rmtree(os.path.join(ROOT_DIR, "build")) if os.path.exists(os.path.join(ROOT_DIR, "build")) else 0
        shutil.rmtree(os.path.join(ROOT_DIR, "dist")) if os.path.exists(os.path.join(ROOT_DIR, "dist")) else 0

class install(setuptools.command.install.install):
    def run(self):
        super().run()

try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    wheel = None
else:
    class wheel(bdist_wheel):

        def write_wheelfile(self, *args, **kwargs):

            super().write_wheelfile(*args, **kwargs)

def configure_extension_build():

    C = Extension(
            "magma",
            include_dirs=[MAGMA_INCLUDE], 
            library_dirs=[MAGMA_LIB],
            sources=['src/_C.c'],
            language='c++'
        )

    cmdclass = {
        "bdist_wheel": wheel,
        "build_ext": BuildExtension,
        "clean": clean,
        "install": install,
        "sdist": sdist,
    }


    extensions = [C]

    return extensions, cmdclass

if __name__ == "__main__":

    detect_gpu_arch()
    #version, sha = get_version()
    version = '2.9.0'

    print(f"Building wheel {PACKAGE_NAME}-{version}")

    with open("README") as f:
        readme = f.read()

    extensions, cmdclass = configure_extension_build()

    dist = Distribution()
    dist.script_name = os.path.basename(sys.argv[0])
    dist.script_args = sys.argv[1:]
    try:
        dist.parse_command_line()
    except setuptools.distutils.errors.DistutilsArgError as e:
        print(e)
        sys.exit(1)

    setup(
        name=PACKAGE_NAME,
        version=version,
        author="ICL ",
        author_email="findtheemaillater@icl.edu",
        url="https://github.com/icl-utk-edu/magma/tree/master",
        description="",
        long_description=readme,
        long_description_content_type="text/markdown",
        license="BSD-3-Clause",
        packages=find_packages(),
        package_data={PACKAGE_NAME: ["lib/libmagma.so", "include/*.h"]},
        package_dir={'': '.'}, 
        extras_require={
        },
        ext_modules=extensions,
        include_package_data=True,
        python_requires=">=3.9",
        cmdclass=cmdclass,
    )