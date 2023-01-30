from setuptools import setup, find_packages

long_descriptions = open('README.md', encoding='utf-8').read()

setup(name="auxeticmop",
      version='1.0.2',
      description="A package for finding meta-material structure using ABAQUS and MOP evolutionary algorithm approaches.",
      author='Seongbin Choi',
      author_email='dcas@naver.com',
      url='https://github.com/c0sogi/AuxeticMOP-with-ABAQUS',
      long_description=long_descriptions,
      long_description_content_type='text/markdown',
      license='MIT',
      python_requires='>=3.6,<3.11',
      install_requires=['numpy', 'numba', 'aiofiles', 'matplotlib', 'scipy', 'dataclasses'],
      packages=find_packages(),
      package_data={"": ["sample_scripts/sample_data/*"]},
      zip_safe=False,
      include_package_data=True,
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Microsoft :: Windows",
      ],
      )
