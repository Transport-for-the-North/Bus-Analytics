# Bus-Analytics

Transport for the North's (TfN) bus-analytics toolset for performing analysis on
GTFS data.

## Scheduler

The scheduler is designed to bring together all of TfN's bus-analytics toolsets. This
scheduler requires [OTP4GB-py repository](https://github.com/transport-for-the-north/otp4gb-py/)
to be cloned and setup on the machine and any of its requirements to be installed in the
scheduler's environment.

The location of the local OTP4GB-py repository needs to be provided as the environment variable
`OTP4GB_REPO_FOLDER` but by default has the value "../otp4gb-py".
