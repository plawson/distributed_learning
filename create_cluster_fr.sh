#!/usr/bin/env bash

aws emr create-cluster \
    --applications Name=Ganglia Name=Spark Name=Zeppelin \
    --ec2-attributes '{"KeyName":"plawson-key-pair-eu-west-3","InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"subnet-e7efd1ad","EmrManagedSlaveSecurityGroup":"sg-7e16c417","EmrManagedMasterSecurityGroup":"sg-83e93bea"}' \
    --service-role EMR_DefaultRole \
    --enable-debugging \
    --release-label emr-5.11.0 \
    --log-uri 's3n://aws-logs-641700350459-eu-west-3/elasticmapreduce/' \
    --name 'Cluster OC project 2' \
    --instance-groups '[{"InstanceCount":4,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":1}]},"InstanceGroupType":"CORE","InstanceType":"c5.xlarge","Name":"Core Instance Group"},{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":1}]},"InstanceGroupType":"MASTER","InstanceType":"c5.xlarge","Name":"Master Instance Group"}]' \
    --configurations file://./emr_config.json \ \
    --scale-down-behavior TERMINATE_AT_TASK_COMPLETION \
    --region eu-west-3 \
     --bootstrap-action Path=s3://oc-plawson-fr/bootstrap-emr.sh