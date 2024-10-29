let socket = new WebSocket("ws://localhost:8000/events");

socket.onmessage = function(event) {
  const currentSlide = Reveal.getCurrentSlide();
  switch(event.data){
    case "right": //swipe_right
      console.log("received 'right' event");
      Reveal.left();
      break;
    case "left": //swipe_left
      console.log("received 'left' event");
      Reveal.right();
      break;
    case "rotate_right": // rotate_right
      console.log("received 'rotate' event");
       rotate(currentSlide, 90);  // defined in helper_methods.js
      break;
    case "rotate_left": //rotate_left
      console.log("received 'rotate' event");
      rotate(currentSlide, -90);  // defined in helper_methods.js
      break;
    case "zoom_in": //spread
      console.log("received 'zoom_in' event");

      // increases zoom by 10%
      zoom(10); // `zoom()` is defined in helper_methods.js
      break;
    case "zoom_out": //pinch
      console.log("received 'zoom_out' event");

      // decreases zoom by 10%
      zoom(-10); // `zoom()` is defined in helper_methods.js
      break;
    case "overview": //flip_table
      console.log("received 'overview' event");
      Reveal.toggleOverview();
      break;
    case "up": //up
      console.log("received 'up' event");
      Reveal.down();
      break;
    case "down": //down
      console.log("received 'down' event");
      Reveal.up();
      break;
    case 'jump_end': //spin
      console.log("received 'jump_end' event");
      Reveal.slide(8);
      break;
    default:
      console.debug(`unknown message received from server: ${event.data}`);
  }
};
