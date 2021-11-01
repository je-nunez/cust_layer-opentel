# About

Deeplearning4j custom neural layer instrumented with OpenTelemetry. This
application uses this custom layer in a MNIST training and evaluation, besides
the other tests on the custom layer provided by the documentation.

Note: the essential custom layer (without OTEL) is
[CustomLayerImplementation](https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/features/customizingdl4j/layers/layer/CustomLayerImplementation.java).

# WIP

This project is a *work in progress*. The implementation is *incomplete*
and subject to change. The documentation can be inaccurate.

# How to compile

To compile:

      mvn clean compile package

# How to run

It uses auto-configuration, so the `OTEL_TRACES_*` environment variables (or
equivalent command-line options `-Dotel.*`) need to be accordingly, mainly the
`OTEL_TRACES_EXPORTER` environment variable (or `-Dotel.traces.exporter=...`
command-line option). For example, to test with a simple standard-error dump:

      export OTEL_TRACES_EXPORTER=logging
      export OTEL_TRACES_SAMPLER=always_on
      my_jar_file=target/custlayer-otel-1.0.0-SNAPSHOT-shaded.jar
      my_java_log_prop_file=target/logging.properties
        
      java  -Djava.util.logging.config.file=${my_java_log_prop_file} \
            -cp "${my_jar_file}"   CustomLayerUsageEx

(You may change the `logging.properties` file provided in order to change the
destination of the dump and/or its format.)

The result is like (where it is seen the forward and backpropagation passes in
the custom neural layer during `training=true`, and at the end the test dataset
evaluation with `training=false`):

      # Note: this is training
      <time-stamp> [INFO] 'activate' : <a-trace.id> <a-span.id> INTERNAL
             [tracer: CustomLayer:]
             AttributesMap{data={iterationCount=15, training=true, epochCount=0, thread.id=1}, ...}
      <time-stamp> [INFO] 'backpropGradient' : <a-trace.id> <a-span.id> INTERNAL
             [tracer: CustomLayer:]
             AttributesMap{data={iterationCount=15, epochCount=0, thread.id=1}, capacity=10, totalAddedValues=3}
      ...
      # Note: this is evaluation
      <time-stamp> [INFO] 'activate' : <a-trace.id> <a-span.id> INTERNAL
             [tracer: CustomLayer:]
             AttributesMap{data={iterationCount=938, training=false, epochCount=1, thread.id=1}, ...}

