# Bus-Analytics

Transport for the North's (TfN) bus-analytics toolset for performing analysis on
GTFS data.

## Scheduler

The scheduler is designed to bring together all of TfN's bus-analytics toolsets. This
scheduler requires [OTP4GB-py repository](https://github.com/transport-for-the-north/otp4gb-py/)
to be cloned and setup on the machine and any of its requirements to be installed in the
scheduler's environment.

### Setup

- Clone and setup [OTP4GB-py repository](https://github.com/transport-for-the-north/otp4gb-py/)
- Create environment and install basic requirements ([requirements.txt](requirements.txt) and
  [requirements_dev.txt](requirements_dev.txt))
- Install requirements only available on pip ([requirements_pip.txt](requirements_pip.txt))
- Install additional OTP4GB requirements, from otp4gb-py repository (`otp4gb-py/requirements.txt`)
- Set environment variable (`OTP4GB_REPO_FOLDER`) pointing to the local otp4gb-py repository
  folder, this defaults to `../otp4gb-py`

- Setup symbolic links to OTP folders within Bus-analytics using,
  `mklink /D folder ..\otp4gb-py\folder`, for the following folders:
  - bin
  - config
  - assets
