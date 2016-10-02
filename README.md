# General CNN player
**Convolutional neural network that solves OpenAI gym environments from raw pixel data via Q-learning**  
The code takes inspiration from the (previously published) CNN for PyGame environments. Original repository can be found [here][5].

### Quick start
Download the code, assign to game_name the name of environment you wish to run, and let the script learn how to solve it.  
Note the code only works for environments with discrete action space and 3-dimensional observation space representing the raw pixel data of the screen.  

Results can be found on my [OpenAI page][4]

### Requirements
* **Python 3**. I recommend this version as it's the only one I found compatible with the below libraries;
* **PyGame**, I used version 1.9.2a0. Download it from [here][1];
* **TensorFlow**, I only managed to install it on my Mac. Download it from [here][2];
* **Gym**, open-source collection of test problems for reinforcement learning algorithms. Details on how to download it can be found [here][3]. 

[1]: http://www.pygame.org/download.shtml 
[2]: https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html 
[3]: https://gym.openai.com/docs
[4]: https://gym.openai.com/users/FlankMe
[5]: https://github.com/FlankMe/player-ConvNN
