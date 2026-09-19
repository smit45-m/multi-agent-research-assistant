# Clean Energy & Battery Technology: Technical Reference

## Solid-State Lithium-Metal Batteries

Solid-state batteries replace flammable liquid electrolytes with a solid
electrolyte — ceramic (garnet LLZO, sulfide argyrodite) or polymer — paired
with a lithium-metal anode. Energy density targets of 400-500 Wh/kg exceed
conventional lithium-ion's 250-300 Wh/kg because the metal anode stores far
more lithium per gram than graphite. The central failure mode is dendrite
growth; dendrite suppression relies on high shear-modulus ceramics, a
ceramic separator layer, and stack pressure. Solid electrolytes also improve
thermal stability, tolerating temperatures that would trigger runaway in
liquid cells. Remaining challenges are interfacial resistance, manufacturing
cost, and maintaining contact through volume change during cycling.

## Perovskite-Silicon Tandem Solar Cells

Tandem cells stack a wide-bandgap perovskite top cell over a silicon bottom
cell, harvesting the spectrum more efficiently than either alone. Bandgap
tuning of the perovskite (1.65-1.70 eV via halide composition) optimizes
current matching. Tandems overcome the single-junction Shockley-Queisser
limit of about 33 percent; certified tandem power conversion efficiencies
have exceeded 33 percent versus about 27 percent for the best silicon.
Commercial viability hinges on the degradation rate: halide segregation
under illumination and moisture stability of the perovskite layer demand
encapsulation and compositional engineering (formamidinium-cesium blends,
2D capping layers). Efficiency percentage retention after damp-heat and
UV-soak testing is the key qualification metric.

## PEM Electrolyzers for Green Hydrogen

Proton exchange membrane electrolyzers split water using a solid polymer
membrane, an iridium catalyst at the oxygen-evolving anode and platinum at
the cathode. They operate at high current density (2-4 A/cm2), respond in
seconds — ideal for coupling to intermittent renewables — and deliver
hydrogen at 30+ bar. System efficiency is expressed as kWh per kg of
hydrogen: about 50-55 kWh/kg today against a thermodynamic floor near 39.4
kWh/kg, driven by cell voltage overpotentials. Cost roadmaps focus on
cutting iridium loading (scarce supply) and slowing membrane degradation
from radical attack, which thins the membrane and raises crossover risk.

## Flow Batteries for Grid Storage

Redox flow batteries decouple power (stack size) from energy (tank volume),
suiting 4-12 hour grid storage. Vanadium redox systems use the same element
in both tanks, so membrane crossover causes capacity loss but not permanent
contamination; they demonstrate cycle life beyond 15,000 cycles and
round-trip efficiency of 70-80 percent. Iron-chromium chemistry offers much
lower electrolyte cost from abundant materials but suffers lower voltage,
hydrogen evolution side reactions, and worse crossover behavior. Levelized
cost per MWh capacity favors flow batteries over lithium-ion as discharge
duration grows, because adding energy means adding cheap electrolyte rather
than full cells.

## Sodium-Ion Batteries

Sodium-ion cells substitute abundant sodium for lithium, using cathodes such
as Prussian blue analogues and layered oxide materials with hard-carbon
anodes. Raw material abundance (sodium, iron, manganese) removes lithium,
cobalt, and often copper from the bill of materials, cutting cost and supply
risk. Energy density trails lithium iron phosphate (120-160 Wh/kg), but
sodium cells show strong capacity retention over thousands of cycles and
notably better sub-zero performance — some chemistries retain over 80
percent capacity at -20 C — plus safe transport at zero volts. They target
stationary storage and entry-level vehicles where cost per kWh beats energy
density.

## Thermal Runaway Mitigation in EV Packs

Thermal runaway propagation is contained through pack-level engineering:
phase change materials absorb heat via latent heat of fusion, flattening
cell temperature spikes; propagation barrier materials (aerogel sheets, mica)
insulate neighboring cells; and increased cell spacing with venting channels
directs hot gases away. Flame retardancy additives and coatings slow
ignition of plastics, while pyrolysis of electrolyte and separator is the
main fuel source once a cell vents. Standards such as GB 38031 require no
pack-level fire for a defined time after single-cell runaway, pushing
designs toward early detection (gas, pressure sensors) plus barrier and
venting strategies validated by nail-penetration and heater-trigger tests.

## Graphene Supercapacitors

Supercapacitors store charge in the electrochemical double layer rather than
through faradaic reactions, giving power density in kW/kg and million-cycle
lifetimes. Graphene electrodes offer specific surface area up to 2630 m2/g;
practical specific capacitance reaches 150-300 farad per gram in aqueous
electrolytes. Performance depends on maintaining ion-accessible pores
(preventing sheet restacking) and minimizing equivalent series resistance
(ESR), which sets the charge discharge rate and peak power. Energy density
remains 5-15 Wh/kg — an order below batteries — so supercapacitors serve
regenerative braking, grid frequency response, and hybrid packs where rate
capability matters more than stored energy.

## Small Modular Reactors (SMR)

SMRs are nuclear reactors under roughly 300 megawatt electrical, factory
built and passively safe. Passive safety cooling removes decay heat without
pumps or operator action: natural circulation loops, gravity feed water
tanks, and containment vessel designs submerged in below-grade pools that
conduct heat to the environment indefinitely. NuScale's design licenses
77 MWe modules whose decay heat removal works by boiling water off the
containment exterior. Economic viability depends on factory series
production offsetting lost economies of scale, and on load-following
capability complementing renewables.

## Direct Air Capture (DAC)

Direct air capture extracts CO2 at 420 ppm concentration, paying a steep
thermodynamic energy penalty: the theoretical minimum is about 20 kJ per
mole CO2, but real systems require 5-10x that once sorbent regeneration is
included. Solid sorbent systems use amine functionalization on porous
supports, capturing CO2 at ambient temperature and releasing it via
temperature-vacuum swing — vacuum desorption at 80-120 C — with thermal
energy input of 1500-2500 kWh per tonne. Liquid hydroxide systems (as
mapped on a Mollier diagram for the steam cycle) regenerate at 900 C in a
calciner, favoring integration with cheap heat. Costs of 400-600 dollars
per tonne today must fall below 200 for gigatonne relevance.

## Solid Oxide Fuel Cells (SOFC)

SOFCs convert fuel to electricity electrochemically at 600-850 C using a
ceramic electrolyte, a cermet anode (nickel-YSZ), and perovskite cathodes,
achieving 55-65 percent electrical efficiency on natural gas. The operating
temperature enables internal reforming but drives degradation mechanisms:
interconnect oxidation raises ohmic resistance; chromium poisoning —
volatile Cr species from stainless interconnects depositing on cathode
sites — suppresses oxygen reduction; nickel coarsening degrades the anode;
and thermal cycling cracks seals through expansion mismatch. Degradation
rates below 0.5 percent per 1000 hours over 40,000+ hour lifetimes are the
commercial benchmark, addressed with coated interconnects and cathode
compositions resistant to Cr uptake.
