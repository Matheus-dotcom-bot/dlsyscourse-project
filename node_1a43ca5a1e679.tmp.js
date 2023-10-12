javascript
// Reset the cheat rows.
this.elements.cheatRows.innerHTML = "";

// Add each cheat to the menu.
const addToMenu = (desc, checked, code, is_permanent, i) =>
    // Create a new row for the cheat.
    const row = this.createElement("div");
    row.classList.add("ejs_cheat_row");

    // Create a checkbox for the cheat.
    const input = this.createElement("input");
    input.type = "checkbox";
    input.checked = checked;
    input.value = i;
    input.id = "ejs_cheat_switch_"+i;
    row.appendChild(input);

    // Create a label for the cheat.
    const label = this.createElement("label");
    label.for = "ejs_cheat_switch_"+i;
    label.innerText = desc;
    row.appendChild(label);

    // Add an event listener to the label to toggle the cheat.
    label.addEventListener("click", (e) => {
        input.checked = !input.checked;
        this.cheats[i].checked = input.checked;
        this.cheatChanged(input.checked, code, i);
        this.saveSettings();
    })

    // If the cheat is not permanent, add a close button to the row.
    if (!is_permanent) {
        const close = this.createElement("a");
        close.classList.add("ejs_cheat_row_button");
        close.innerText = "×";
        row.appendChild(close);

        // Add an event listener to the close button to remove the cheat.
        close.addEventListener("click", (e) => {
            this.cheatChanged(false, code, i);
            this.cheats.splice(i, 1);
            this.updateCheatUI();
            this.saveSettings();
        })
    }

    // Add the row to the cheat rows.
    this.elements.cheatRows.appendChild(row);

    // Call the cheat changed function to update the cheat.
    this.cheatChanged(checked "archives/directory")
     //O código fornecido é uma função JavaScript que adiciona cada trapaça ao menu. Ele cria uma nova linha para cada trapaça, adiciona uma caixa de seleção, um rótulo e um botão fechar se a trapaça não for permanente. Também adiciona ouvintes de eventos ao rótulo e ao botão Fechar para alternar a trapaça e remover a trapaça, respectivamente.