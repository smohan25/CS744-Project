"""ubuntu 18 profile for cs 744 Fall 2020 with 8 connected vm nodes"""

# Import the Portal object.
import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as rspec #pg
# Import the Emulab specific extensions.
import geni.rspec.emulab as emulab

# Create a portal object,
pc = portal.Context()

# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()

node0 = request.XenVM("node0")
node0.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD'
node0.Site("Site 1")
#node0.routable_control_ip = "false"
node0.cores = 5
node0.ram = 16384
node0.disk = 20
# node0.hardware_type = "c220g2"

node1 = request.XenVM("node1")
node1.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD'
node1.Site("Site 1")
#node1.routable_control_ip = "false"
node1.cores = 5
node1.disk = 20
node1.ram = 16384
# node1.hardware_type = "c220g2"

node2 = request.XenVM("node2")
node2.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD'
node2.Site("Site 1")
#node2.routable_control_ip = "false"
node2.cores = 5
node2.disk = 20
node2.ram = 16384
# node2.hardware_type = "c220g2"

node3 = request.XenVM("node3")
node3.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD'
node3.Site("Site 1")
#node2.routable_control_ip = "false"
node3.cores = 5
node3.disk = 20
node3.ram = 16384
# node2.hardware_type = "c220g2"

node4 = request.XenVM("node4")
node4.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD'
node4.Site("Site 1")
#node2.routable_control_ip = "false"
node4.cores = 5
node4.disk = 20
node4.ram = 16384
# node2.hardware_type = "c220g2"

node5 = request.XenVM("node5")
node5.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD'
node5.Site("Site 1")
#node2.routable_control_ip = "false"
node5.cores = 5
node5.disk = 20
node5.ram = 16384
# node2.hardware_type = "c220g2"

node6 = request.XenVM("node6")
node6.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD'
node6.Site("Site 1")
#node2.routable_control_ip = "false"
node6.cores = 5
node6.disk = 20
node6.ram = 16384
# node2.hardware_type = "c220g2"

node7 = request.XenVM("node7")
node7.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD'
node7.Site("Site 1")
#node2.routable_control_ip = "false"
node7.cores = 5
node7.disk = 20
node7.ram = 16384
# node2.hardware_type = "c220g2"

link1 = request.Link(members = [node0, node1, node2, node3, node4, node5, node6, node7])
link1.best_effort = True

#bs = node0.Blockstore("bs", "/test-data")
#bs.dataset = "urn:publicid:IDN+wisc.cloudlab.us:cs744-s19-pg0+imdataset+enwiki-data"
#bs.readonly = True

# # # Special attributes for this link that we must use.
# fslink.best_effort = True
# fslink.vlan_tagging = True

# Print the generated rspec
pc.printRequestRSpec(request)
