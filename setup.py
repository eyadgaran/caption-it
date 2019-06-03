from setuptools import setup, find_packages

setup(name='captioner',
      version='1.0',
      description='CaptionIt',
      url='https://github.com/eyadgaran/caption-it.git',
      author='Elisha Yadgaran',
      author_email='elishay@alum.mit.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'flask',
          'sqlalchemy',
          'sqlalchemy-mixins',
          'simpleml[all]',
          'opencv-python',
          'tqdm',
          'pandas',
          'numpy',
          'requests',
          'keras',
          'imagehash',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False,
      include_package_data=True,
      package_data={'': ['templates/*', 'static/*']},
    )
