export interface CounterController {
  increment: () => void;
  decrement: () => void;
  set_counter: (value: number) => void;
  reset_counter: () => void;
  getValue: () => number;
}

export function setupCounter(element: HTMLButtonElement): CounterController {
  let counter = 0;

  const updateDisplay = () => {
    element.innerHTML = `count is ${counter}`;
  };

  const increment = () => {
    counter++;
    updateDisplay();
  };

  const decrement = () => {
    counter--;
    updateDisplay();
  };

  const set_counter = (value: number) => {
    counter = value;
    updateDisplay();
  };

  const reset_counter = () => {
    counter = 0;
    updateDisplay();
  };

  const getValue = () => counter;

  element.addEventListener("click", increment);
  updateDisplay();

  return {
    increment,
    decrement,
    set_counter,
    reset_counter,
    getValue,
  };
}
