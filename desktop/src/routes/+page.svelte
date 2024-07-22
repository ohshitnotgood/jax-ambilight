 <script lang="ts">
    import * as IPCClient from "../components/ipc";
    import * as utils from "../components/utils"

    const availableHeightZones = [2, 3, 4, 5, 6];
    const availableWidthZones = [4, 5, 6, 7, 8];

    let nWidthZones = 6;
    let nHeightZones = 4;
    let monitor = "Select monitor";        
    let colourList: string[][] = [];

    /**
     * Returns a list of available monitors
     */
    async function getMonitors() {
        return ["Monitor 1", "Monitor 2"];
    }

    function startUpRoutine() {
        IPCClient.createIPCClient()
        IPCClient.onMessageReceived((data: string) => {
            colourList = utils.parseBackendData(data)
            IPCClient.sendMessage("ack_ok")
        })
        IPCClient.sendMessage("ack_ok")
    }


    function getRandomRGBColour() {
        return `background-color: rgb(${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)})`
    }

    function getTestColours(widthZones: number, heightZones: number) {
        let widthList = [];
        for (let i = 0; i < widthZones; i++) {
            widthList.push(getRandomRGBColour());
        }
        colourList.push(widthList);

        let heightList = [];
        for (let i = 0; i < heightZones; i++) {
            heightList.push(getRandomRGBColour());
        }
        colourList.push(heightList);

        heightList = [];
        for (let i = 0; i < heightZones; i++) {
            heightList.push(getRandomRGBColour());
        }
        colourList.push(heightList);

        widthList = [];
        for (let i = 0; i < widthZones; i++) {
            widthList.push(getRandomRGBColour());
        }
        colourList.push(widthList);

        return colourList;
    }

    colourList = getTestColours(8, 4)
</script>

<main class="grid place-content-center">

    <button on:click={startUpRoutine}>
        Connect to server
    </button>

    <controls class="grid grid-cols-[1fr_2fr] place-content-center">
        <labels class="block text-right p-4">
            <div class="pt-1.5 pb-1">Select monitor</div>
            <div class="pt-1.5 pb-1">Width zones</div>
            <div class="pt-1.5 pb-1">Height zones</div>
        </labels>

        <selectors class="block py-4">
            <monitor-selector class="block text-left">
                {#await getMonitors()}
                    Loading monitors
                {:then monitors}
                    <select
                        bind:value={monitor}
                        class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-md block px-4 pt-1.5 pb-1"
                    >
                        {#each monitors as each_monitor}
                            <option>{each_monitor}</option>
                        {/each}
                    </select>
                {/await}
            </monitor-selector>

            <hor-selector class="block pt-1.5 pb-1">
                <select
                    bind:value={nWidthZones}
                    class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-md block px-4 pt-1.5 pb-1"
                >
                    {#each availableWidthZones as each_zone}
                        <option>{each_zone}</option>
                    {/each}
                </select>
            </hor-selector>

            <vert-selector class="block pt-1 pb-1">
                <select
                    bind:value={nHeightZones}
                    class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-md block px-4 pt-1.5 pb-1"
                >
                    {#each availableHeightZones as each_zone}
                        <option>{each_zone}</option>
                    {/each}
                </select>
            </vert-selector>
        </selectors>
    </controls>

    <preview class="block p-4">
        <div class="flex mx-[50px] w-[384px]">
            {#each { length: nWidthZones } as _, i}
                <width-zone style={colourList[0][i]}/>
            {/each}
        </div>
        <div class="flex justify-between">
            <div class="flex flex-col h-[218px]">
                {#each { length: nHeightZones } as _, i}
                    <height-zone style={colourList[1][i]} />
                {/each}
            </div>
            <div class="flex flex-col h-[218px]">
                {#each { length: nHeightZones } as _, i}
                    <height-zone style={colourList[2][i]}/>
                {/each}
            </div>
        </div>
        <div class="flex mx-[50px] w-[384px]">
            {#each { length: nWidthZones } as _, i}
                <width-zone style={colourList[3][i]}/>
            {/each}
        </div>
    </preview>
</main>

<style>
    width-zone {
        @apply bg-yellow-400 w-full h-[50px];
    }

    height-zone {
        @apply bg-red-300 w-[50px] h-full;
    }
</style>
