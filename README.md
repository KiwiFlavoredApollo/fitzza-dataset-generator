# README

Musinsa를 크롤링했습니다.

## background-remover.py

- 필요에 따라 `INPUT`과 `OUTPUT`의 값을 수정해주어야 합니다.
- GPU를 사용하는 것을 추천합니다.
- GPU를 활용하기 위해서는 CUDA와 cuDNN을 설치해야 합니다.
- cuDNN의 경로를 환경변수에 추가하는 작업이 필요했습니다.
- RTX 3060 12GB를 사용했을 때 이미지당 6초 정도 걸립니다. 