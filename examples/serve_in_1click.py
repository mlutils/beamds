from beam.misc.fake import BeamFakeAlg

fake_alg = BeamFakeAlg(sleep_time=1)

print(fake_alg.run(123))

# from beam.serve import beam_server
# beam_server(fake_alg)



# from beam.auto import AutoBeam
# AutoBeam.to_docker(fake_alg)


from beam.orchestration.cluster import HTTPServeCluster


HTTPServeCluster.from_algorithm(fake_alg)