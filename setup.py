from os import path, getcwd

from setuptools import setup, find_packages

package_name = 'spark_feature_selector'

try:
    with open(path.join(getcwd(), 'VERSION')) as version_file:
        version = version_file.read().strip()
except IOError:
    raise


def parse_requirements(file):
    with open(file, "r") as fs:
        return [r for r in fs.read().splitlines() if
                (len(r.strip()) > 0 and not r.strip().startswith("#") and not r.strip().startswith("--"))]


requirements = parse_requirements('requirements.txt')
test_requirements = parse_requirements('requirements-test.txt')

setup(name=package_name,
      version=version,
      license='Apache License 2.0',
      description='Library for smart AI/ML based feature selection and ranking',
      author='Iman Haji',
      author_email='iman@clarifyhealth.com',
      url='https://github.com/clarifyhealth/spark_feature_selector',
      packages=find_packages(exclude=['scripts']),
      install_requires=requirements,
      tests_require=test_requirements,
      include_package_data=True,
      zip_safe=False)
