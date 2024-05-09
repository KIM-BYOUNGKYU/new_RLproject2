#https://github.com/cmontalvo251/aerospace/blob/main/rockets/rocket_seminar_series/two_stage_rocket_w_aerodynamics.py
* action space
- 5개의 엔진 축에 수직방향 각속도 (-30,0,30)
- 5개의 엔진 축방향 각속도 (-30,0,30)
- 5개의 엔진 추력(min~max) => continuous action space
- 3개의 엔진 축에 수직방향 각속도 (-30,0,30)
- 3개의 엔진 축방향 각속도 (-30,0,30)
- 3개의 엔진 추력(min~max) => continuous action space
- 2개의 분리(버튼)
=> 총 26개의 값을 return


* state space
- (x, y, z 좌표)    list:3개
- (x, y, z 속도)    list:3개
- (회전 각도)        list:3개
- (회전 각속도)     list:3개
- 현재 연료 질량
- 현재 stage
- 선체를 기준으로 노즐 각도 (0theta0, 0theta1), (1theta0, 1theta1), (2theta0, 2theta1), (3theta0, 3theta1), ...) list의 list: 2*8개
    theta0 = 0~2pi
    theta1 = 0~max
=> 30개의 state를 저장