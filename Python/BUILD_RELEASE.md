# Notes on Building Releases

## Building a PyPi release:

In order to build a release follow these steps:

 1. In your local repository go to the folder just above the `package` folder and do,
	```
	$ R CMD build package
	$ R CMD check --as-cran popsom7_xyz.tar.gz
	```
	where `xyz` is the version number given in the DESCRIPTION file.  Note: This builds a folder called `popsom7.Rcheck`.

2. If the check passes, submit tarball `popsom7_xyz.tar.gz` to `https://cran.r-project.org/submit.html`
