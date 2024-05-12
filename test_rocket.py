import rocket

falcon9 = rocket.Rocket()
action = []
for i in range(5):
    action.append(0)
for i in range(5):
    action.append(1000)
for i in range(5):
    action.append(falcon9.max_thrust[0])

for i in range(9):
    action.append(0)

action.append(0)


falcon9.step(action)
for i in range(100000):
    falcon9.step(action)

falcon9.show_path_from_state_buffer()
falcon9.animate_trajectory()